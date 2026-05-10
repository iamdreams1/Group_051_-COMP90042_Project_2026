# COMP90042 Project — Re-plan with FEVER-pretrained Models

## 总体设计哲学

**核心洞见**：本项目的任务格式（claim + 多 evidence → 4 类标签）几乎是 FEVER 数据集的同构问题。一旦允许使用 FEVER 预训练 checkpoint，整个分类 pipeline 应该**围绕 NLI 范式重构**，而不是把 evidence 当成 context concat 给一个通用分类器。

具体三个范式转变：

1. **分类 backbone**：通用 MLM (deberta-v3-base) → FEVER 预训练 NLI (MoritzLaurer 系列)
2. **评分粒度**：`(claim, concat-of-5-evidence) → 4 类` → `(claim, single evidence) → 3 类，再聚合 → 4 类`
3. **DISPUTED 处理**：从普通的第 4 类 → 显式建模为"多 evidence 给出冲突 NLI 信号"

---

## Section 1: Data Preprocessing

### 1.1 加载与缓存（保留现有方案）
- 读取四份 json：`train-claims`, `dev-claims`, `test-claims-unlabelled`, `dev-claims-baseline`
- `evidence.json` (174 MB) 转 pickle，加速重启加载
- 构建 `{evi_id → text}` dict 与 `[evi_id]` 有序列表（保证 BM25 索引位置和 dense embedding 位置一致）

### 1.2 数据清洗（新增）
| 项目 | 处理方式 | 理由 |
|---|---|---|
| Evidence > 500 词 | Truncate 到 256 词 | 避免 reranker / 分类器 max_len 截断不一致 |
| Evidence < 3 词 | 标记为低质量但保留 | 不能丢弃，eval 时仍可能命中 gold |
| Claim 文本 | 仅保留原文 | NLI 预训练模型对原文格式敏感，过度归一化反而掉点 |

### 1.3 EDA（保留 + 强化）
- Claim 长度分布、4 类标签分布、每条 claim 的 gold evidence 数（已有）
- **新增**：Evidence 长度分布（决定 `max_seq_length`，预计 256 够用）
- **新增**：DISPUTED 类样本逐条审视——这一类 train 上只有几十条，必须人工看一遍才能设计正确的 loss 加权

### 1.4 训练对构造（重要的新增模块）

**NLI 风格的分类训练对**（替代之前 5-evi concat 输入）：
- 每条 claim 拆成 N 个 `(claim, gold_evidence_i)` 对
- 标签继承 claim 的 4 类标签
- train 上约 1228 × avg(3) ≈ **3700 条 NLI 训练样本**（远多于原来 1228 条）

**Reranker 训练对**（仅做诊断 / ablation 用）：
- (claim, gold) = 1 / (claim, hard_neg from hybrid) = 0
- 实际**不再用于微调**（直接用 bge-reranker zero-shot）

**DISPUTED 类平衡**：
- 训练时 DISPUTED 上采样 3–5x
- Loss 中 class weight ≈ 6x（按 inverse frequency）

---

## Section 2: Model Implementation

### 2.1 检索阶段（M2-a）

| 组件 | 模型 | 大小 | 作用 |
|---|---|---|---|
| Sparse | BM25Okapi (NLTK 分词 + stopwords + alpha-only) | - | 字面匹配召回 |
| Dense | `BAAI/bge-base-en-v1.5` | 109M, 768d | 语义召回 |
| Fusion | BM25 Top-200 ∪ Dense Top-200 | - | 互补 (字面 + 语义) |

**预期**：Hybrid Recall@400 ≥ 78%（现有方案 ≈ 73.7%，主要靠 dense 升级）

### 2.2 重排阶段（M2-b）

`BAAI/bge-reranker-base` (278M) zero-shot

不再尝试微调，吸取实验 A 教训。重排后取 Top-5 喂分类器。

**预期**：F@5 ≥ 0.22（现有 baseline reranker 0.188）

### 2.3 分类阶段（M3）—— 真正的核心

**Backbone**：`MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (435M)
- 已在 MNLI + FEVER + ANLI + LingNLI + WANLI 上预训练
- 原生 3 类 head: entailment / neutral / contradiction
- T4 fp16 + grad_accum=8 可训

如果 T4 显存吃紧，退而求其次：`MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (184M)，drop-in 替换原来的 deberta-v3-base，几乎不增加显存。

#### 设计 A（首选）：Per-evidence NLI 评分 + 学习型聚合

对 claim 的 Top-5 evidence，**逐条**走 backbone：

```
input_i = [CLS] claim [SEP] evidence_i [SEP]
output_i = (p_entail_i, p_neutral_i, p_contradict_i)   # 3-class softmax
```

聚合层：
```
features = [
    max(p_entail), mean(p_entail), std(p_entail),
    max(p_neutral), mean(p_neutral), std(p_neutral),
    max(p_contradict), mean(p_contradict), std(p_contradict),
    top-5 reranker scores,
    n_evi_above_threshold
]   # 维度约 20

logits = MLP(features)   # → 4 类
```

**为什么这样设计有道理**：
- **保留 FEVER 预训练知识**：backbone 输出仍是 3 类 NLI，前几个 epoch 完全冻结
- **DISPUTED 显式可学**：`max(p_entail) 高 AND max(p_contradict) 高` 的模式直接对应 DISPUTED 的语义（不同 evidence 立场冲突），用普通 concat 模型很难学到
- **std 特征捕捉 disagreement**：5 条 evidence 之间的方差是 DISPUTED 的关键信号
- **OOP 干净**：backbone 与 aggregator 解耦，可分别训练

#### 设计 B（消融对照）：4 类头 + concat 多 evidence

把 backbone 的 3 类 head 替换成 4 类 head，输入仍是
`[CLS] claim [SEP] evi_1 [SEP] ... [SEP] evi_5 [SEP]`

这是现有实现的直接迁移，作为 ablation 对照（"我们做了 design A 比 B 好 X%"）。

#### 三阶段训练策略

| 阶段 | 数据 | 可训练参数 | LR | Epoch | 目的 |
|---|---|---|---|---|---|
| 1 | (claim, gold_evi) NLI 对（~3700 条） | 整个 backbone | 1e-5 | 2 | 让模型适应气候 domain，但不破坏 NLI 能力 |
| 2 | reranker Top-5 输出 | **冻结 backbone**，只训 aggregator MLP | 1e-3 | 3 | 学聚合规则（DISPUTED 在这里学到） |
| 3 (可选) | reranker Top-5 输出 | 全部解冻 | 2e-6 | 1 | 联合微调，小步长 |

#### Loss 设计
- Cross-entropy + class weight（按 inverse frequency 归一化使 sum=4）
- Label smoothing = 0.1
- 可选：DISPUTED 用 focal loss（γ=2）单独处理

### 2.4 OOP 类设计（满足项目硬性要求）

| 类 | 父类 | 职责 |
|---|---|---|
| `EvidenceCorpus` | - | 检索器统一接口 |
| `BM25Retriever`, `DenseRetriever`, `HybridRetriever` | `EvidenceCorpus` 子类 | 三种召回策略 |
| `NLIClaimDS` | `torch.utils.data.Dataset` | 生成 (claim, single evidence) NLI 输入 |
| `AggregatorMLP` | `nn.Module` | 把 5 条 evidence 的 NLI 概率聚合为 4 类 logits |
| `FactVerifier` | `nn.Module` | NLI backbone + aggregator 整合 |
| `Trainer` | - | 三阶段训练循环（fp16 / class weight / cosine LR / early stop） |

---

## Section 3: Testing and Evaluation

### 3.1 端到端 predict 函数
输入：`claims = {cid: {...}}`
输出：`{cid: {claim_text, claim_label, evidences[≤5]}}`，符合 eval.py 期望格式

流程：
1. Hybrid 检索 ≤400 候选
2. bge-reranker 重排取 Top-5
3. FactVerifier 对 Top-5 做 per-evi NLI + 聚合 → 4 类预测

### 3.2 Dev 主评估
- 用官方 `eval.py` 算 **Hmean**（最终主指标）
- 同时打印：retrieval Recall@K (K=5/50/200), reranker F@5, classification accuracy, **per-class F1**（特别是 DISPUTED）

### 3.3 消融实验（报告核心）

| 消融组 | 替换 | 预期 ΔHmean | 报告论点 |
|---|---|---|---|
| Sparse-only | 去掉 dense 检索 | -0.05 | Hybrid 必要性 |
| 不重排 | hybrid 原始顺序送分类 | -0.08 | Reranker 必要性 |
| 通用 DeBERTa | 换回 deberta-v3-base | -0.04 ~ -0.06 | FEVER 预训练价值 |
| Concat 输入 (设计 B) | 替代 per-evi 聚合 | -0.03 (主要影响 DISPUTED F1) | 聚合策略价值 |
| 等权 CE loss | 去 class weight | DISPUTED F1 → ~0 | 类别不平衡处理 |
| Top-1 vs Top-5 | 只用最高分 evidence | -0.02 ~ -0.05 | 多 evidence 必要性 |

每组消融都是报告里的一段独立论述。

### 3.4 错误分析（必备章节）
- **4×4 混淆矩阵**：哪些类被错分得最多
- **DISPUTED 案例研究**：手挑 5 条 false negative + 5 条 false positive，看模型为什么错
- **检索失败案例**：hybrid 都救不回来的 claim，分析它们的语言特征（很口语 / 实体 OOV / 数字主导等）
- **Top-K 敏感性**：reranker Top-3 / 5 / 8 对最终 Hmean 的影响曲线

### 3.5 与 baseline 对比
- 在 dev 上跑 `dev-claims-baseline.json` 的 Hmean 作为下界
- 报告"我们 vs baseline" 的提升 + 各阶段贡献分解

### 3.6 最终提交
- 生成 `test-claims-predictions.json`
- **格式严格校验**：153 条 / 标签合法 / evidence 数 ∈ [1, 5]
- 在 README 中写明：使用的开源 checkpoint 全名 + 版本 + 是否微调（reproducibility）

---

## 时间规划（建议）

| 周 | 任务 | 交付物 |
|---|---|---|
| W1 | Section 1 完成 + BM25 baseline | EDA 图 + Recall@200 数字 |
| W2 | Dense (bge-base) + Hybrid + reranker zero-shot | F@5 数字 + cache pickle |
| W3 | 分类器 stage 1（gold NLI 训练） | dev classification acc |
| W4 | 分类器 stage 2-3（aggregator 训练 + 联调） | 完整 Hmean |
| W5 | 消融 + 错误分析 + 报告写作 | PDF 报告 + 提交文件 |

---

## 主要风险与对策

| 风险 | 影响 | 对策 |
|---|---|---|
| FEVER 预训练算"同任务标注泄漏"被扣分 | 高 | **答辩前邮件向老师确认**，不行就退回 MNLI-only 预训练（如 `roberta-large-mnli`） |
| DeBERTa-v3-large (435M) Colab T4 OOM | 中 | 降级到 base 版本 (184M)；性能损失约 1-2% |
| `bge-base-en-v1.5` 编码 1.2M 段超时 | 中 | T4 fp16 batch=128 实测 ~50min；分批跑并 checkpoint |
| Aggregator MLP 在 1228 条 claim 上过拟合 | 中 | 强 dropout (0.5) + early stop on dev；维度只用 ~20 |
| DISPUTED 类样本太少导致 F1 不稳定 | 高 | 上采样 + class weight + focal loss；报告中如实说明该类样本量限制 |

---

## 与原方案的对比（为报告准备）

| 维度 | 原方案 | 新方案 |
|---|---|---|
| 分类 backbone | deberta-v3-base (通用 MLM) | DeBERTa-v3-large/base **+ FEVER pretrain** |
| 输入构造 | `[CLS] claim [SEP] evi_1 ... evi_5 [SEP]` | `[CLS] claim [SEP] evi_i [SEP]` × 5 |
| Evidence 利用 | concat 后单次 forward | 逐条 forward + 学习型聚合 |
| DISPUTED 处理 | 隐式（依赖 class weight） | 显式（依赖 NLI 概率的 disagreement 特征） |
| Reranker | baseline zero-shot | bge-reranker-base zero-shot（更强） |
| Dense retriever | bge-small | bge-base |
| 预期 dev Hmean | ~0.13–0.16 | **~0.18–0.22** |
