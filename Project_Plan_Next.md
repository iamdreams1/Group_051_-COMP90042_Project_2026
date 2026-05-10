# COMP90042 Group_051 — 后续实施计划（baseline → FEVER NLI 升级）

## Context

**为什么写这份计划**：你问"现在到哪一步、后面怎么做"。审计完仓库后发现：

- ✅ M1（数据加载、EDA、evidence 过滤、4,121 NLI 训练对）已完成且 cache 齐全
- ✅ M2-a（BM25 + BGE-base hybrid 检索）已完成，`cache/hybrid_{train,dev,test}.pkl` + FAISS index + 3.75GB embedding 全部就绪
- ❌ Notebook 当前停在 **cell 21**（hybrid 函数验证），cell 23 起 Section 3 / OOP 区域是空 markdown 占位
- ❌ 5/4 那次跑出的 `ckpt/cls_best.pt`、`cache/reranked_*.pkl`、`dev-predictions.json` —— **代码已被你 5/7 重构时删掉，按你确认"只能用现有的"，下面的计划不复用它们**

**路线决策（你已选）**：先把原 PLAN.md 的 baseline（CrossEncoder + DeBERTa-v3-base 4 分类）跑通拿到一个 dev Hmean，再叠加 Project_Replan_FEVER.md 的 FEVER-NLI 升级，**两个版本都进报告做消融对比**。距离 2026-05-22 截止还有 13 天，时间够用但不能浪费。

---

## Phase A — Baseline 跑通（5/9–5/13，5 天）

### A1. M2-a 收尾：Recall@K 评估表（半天）

**为什么**：cell 20 只测了 `"Is global warming real?"` 一条 claim，没有 dev 上的 Recall 数字。报告里那张 "BM25/Dense/Hybrid" 消融表（PLAN.md §3 的核心证据）必须自己算出来。

**做什么**（在 cell 21 之后插入新 cell）：
1. 写 `recall_at_k(set_of_ids, gold)` 工具函数
2. 在 dev 154 条上分别算 BM25-only、BGE-only、Hybrid (∪) 在 K∈{50, 100, 200, 400} 的 Recall
3. 把 `(common_ids, bm25_set, bge_set)` 改成 **并集**（当前 cell 20 用的是交集 `intersection`，与 PLAN.md / 5/4 实测的 0.844 不一致 —— 这是个 bug，要修）
4. 缓存 `hybrid_{train,dev,test}.pkl`：每条 claim 的并集候选 list（≤400），供 reranker 用

**验收**：dev 上 Hybrid Recall@400 应回到 ~0.84 水平。

### A2. M2-b CrossEncoder Reranker（1.5 天）

**做什么**（新增 cell 在 Section 2 末尾）：
1. **训练对构造**：从 `train_pairs` (4,121 条 gold) + 每条 claim 从 hybrid 候选里挑 4 个非 gold 作为 hard negative → 约 9.7K 对（gold 1.0 / neg 0.0）
2. **模型**：`cross-encoder/ms-marco-MiniLM-L-6-v2`，max_length=256，2 epoch，batch=16，T4 约 15 min
3. **重排**：对 dev hybrid 候选打分排序，存 `cache/reranked_{train,dev,test}.pkl`
4. **F@K 调优**：在 dev 上算 F@{3,4,5,6}，选最优 K（多半是 4 或 5）

**关键复用**：`evi_ids_filtered` / `evidence_filtered`（cell 11 已建）/ `hybrid_*` cache（A1 产出）

**验收**：dev 至少一个 K 上 F ≥ 0.18（PLAN.md §4 验收线 0.20 略放宽，因为 baseline 不强求）

### A3. M3 DeBERTa 4 分类（baseline 版，2 天）

**做什么**（新增 cell + OOP 类区在 cell 25 之后）：
1. **Dataset 类 `ClaimDS`**：拼 `[CLS] claim [SEP] evi_1 [SEP] ... [SEP] evi_K [SEP]`，max_len=512，K = A2 选出的最优值
2. **模型 `BaselineClassifier`**：`microsoft/deberta-v3-base` + 单层 cls 头（**不要做 gated attention，留给 Phase B 当对照**）
3. **Trainer 类**（OOP 满足项目要求）：fp16 + class weight (inverse freq, sum=4) + label smoothing 0.1 + cosine LR + early stop on dev acc，3 epoch，bs=8
4. **训练数据**：用 gold evidence 训前 2 epoch，最后 1 epoch 切到 reranker Top-K（domain shift 适应）

**验收**：dev_acc ≥ 0.45（随机 0.25）。OOM 时 `bs=4 + grad_accum=2`。

### A4. M4 端到端预测 + eval.py（半天）

**做什么**：
1. `predict(claims, reranked, k) -> {cid: {claim_text, claim_label, evidences}}` 函数
2. 在 dev 上跑一遍生成 `dev-predictions.json`
3. 调用 `eval.py --predictions dev-predictions.json --groundtruth data/dev-claims.json` 拿到 baseline 的 **F / A / Hmean**

**验收**：Hmean ≥ 0.13（PLAN.md 期望 0.15，给 baseline 留余地）。

**这就是报告里要超越的"我们自己的 baseline"**。

---

## Phase B — FEVER NLI 升级（5/14–5/19，6 天）

走 Project_Replan_FEVER.md 设计 A：per-evidence NLI 评分 + 学习型聚合 MLP。

### B1. 路线合规确认（最先做，半天）

**为什么**：FEVER 预训练 checkpoint 是否算"同任务标注泄漏"是高风险问题。
- **行动**：发邮件给老师 / 在 Discussion 板上问，附上 `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` 的 README 链接，明确写"FEVER 是预训练数据之一，本项目数据未参与该预训练"
- **若不允许**：退回到 `roberta-large-mnli`（仅 MNLI 预训练，绝对安全）

### B2. Reranker 替换为 zero-shot bge-reranker（半天）

**做什么**：
- 把 A2 的微调 CrossEncoder 替换为 `BAAI/bge-reranker-base` zero-shot，对 hybrid 候选直接打分
- 不再训练，省 15 min；存 `reranked_v2_*.pkl`
- 报告里做 "MiniLM finetune vs bge-reranker zero-shot" 的 F@5 对比

**预期**：F@5 从 ~0.18 提到 ~0.22

### B3. NLI Backbone + 聚合 MLP（3 天，最关键）

**做什么**（新增 OOP 类）：

| 类 | 职责 |
|---|---|
| `NLIClaimDS` | 生成 `(claim, single evidence)` 对，标签是 claim 的 4 类（继承） |
| `FactVerifier` | 持有 `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`（OOM 退 `-base-mnli-fever-anli`）；前向时 K 条 evidence 各过一次 backbone，输出 K×3 的 (entail/neutral/contradict) 概率 |
| `AggregatorMLP` | 输入约 20 维特征：`max/mean/std` of 三类概率 + 5 个 reranker 分 + n_evi_above_threshold；输出 4 类 logits |
| `Trainer`（复用 A3 的，扩展三阶段） | 阶段1：(claim, gold_evi) NLI 训整个 backbone，2 epoch，lr=1e-5；阶段2：冻 backbone 训 aggregator，3 epoch，lr=1e-3；阶段3（可选）：全解冻联调 1 epoch，lr=2e-6 |

**DISPUTED 处理**（这是创新点的核心论点）：
- DISPUTED 训练样本上采样 3-5x
- Class weight ≈ 6x（inverse frequency）
- 聚合特征里 `max(p_entail) 高 AND max(p_contradict) 高` 的模式直接对应"证据立场冲突" → DISPUTED 显式可学

**验收**：dev Hmean 比 Phase A baseline 高 ≥ 0.03，DISPUTED per-class F1 > 0（baseline 大概率是 0）

### B4. 端到端 + 消融实验（2 天）

按 Project_Replan_FEVER.md §3.3 跑完整消融：

| 组 | 改动 | 报告论点 |
|---|---|---|
| Sparse-only | 去掉 BGE | Hybrid 必要性 |
| 不重排 | hybrid 直接送分类 | Reranker 必要性 |
| 通用 DeBERTa | 换回 deberta-v3-base | FEVER 预训练价值 |
| Concat 输入 | 用 Phase A 的 baseline 替代 per-evi 聚合 | 聚合策略价值（这就是 Phase A 的天然作用） |
| 等权 CE | 去 class weight | DISPUTED F1 → ~0 |

---

## Phase C — 提交准备（5/20–5/22，3 天）

### C1. Test 集推理 + 格式校验
- 跑 `predict(test, reranked_test, k)` → `test-claims-predictions.json`
- assert 153 条 / 标签合法 / `1 ≤ len(evidences) ≤ 6`

### C2. Notebook 整理
- 完整重跑 ipynb，**保留所有 cell 输出**（评分要看）
- 每个 section 顶部加 markdown 说明（方法 + 为什么）
- 删调试 print

### C3. ACL ≤7 页报告
按 PLAN.md §8.3 + Project_Replan_FEVER.md §3 结构：
1. Introduction：贡献 = Hybrid retrieval + FEVER-NLI per-evidence 聚合 + DISPUTED 显式建模
2. Method：架构图 + Aggregator 公式
3. Experiments：检索消融 / 分类消融 / 端到端 vs `dev-claims-baseline.json`
4. Error Analysis：4×4 混淆矩阵 + 10 条 DISPUTED case study + 检索失败案例
5. Conclusion + Team Contribution

### C4. ZIP 打包
- `Group_051__COMP90042_Project_2026.ipynb`（含输出）
- `test-claims-predictions.json`
- `README.md`
- **不**含 `data/` 和 `ckpt/`

---

## 需要修改的关键文件

| 文件 | 改动 |
|---|---|
| `Group_051__COMP90042_Project_2026.ipynb` cell 21 后 | 插入 A1 Recall@K 评估 + hybrid 并集 cache |
| 同上 cell 23（Section 3）| 插入 A4 端到端 predict + eval.py 调用 |
| 同上 cell 25 后（OOP 区） | 插入 A2/A3/B2/B3 所有类：`CrossEncoderReranker` / `ClaimDS` / `BaselineClassifier` / `NLIClaimDS` / `FactVerifier` / `AggregatorMLP` / `Trainer` |
| `cache/hybrid_*.pkl` | A1 用并集重新生成（当前是 cell 20 测试时生成的交集，需修） |

## 可复用的现有代码（不要重造轮子）

| 已有 | 在哪 | 怎么用 |
|---|---|---|
| `nltk_tokenize()` | cell 15 | reranker 训练对、A2 兜底分词都直接调 |
| `evidence_filtered`, `evi_ids_filtered` | cell 11 | 所有下游 evidence 文本来源 |
| BM25 索引 `bm25` | cell 16 | A1 直接复用 |
| BGE FAISS index `index` + `embeddings` | cell 19 | A1 直接复用 |
| `get_hybrid_top_n()` | cell 20 | **要修**：把 `intersection` 改成 `union`；当前实现错了 |
| `train_pairs` (4,121 条 gold) | cell 12 | A2 的正样本来源；B3 NLI 数据集的种子 |

## 验证方式（端到端跑一遍）

完成 Phase A 后：
```bash
python eval.py --predictions dev-predictions.json --groundtruth data/dev-claims.json
```
应输出 F、A、Hmean 三个数字。Hmean ≥ 0.13 即 baseline 通过。

完成 Phase B 后：再跑一次同命令，Hmean 应比 A 阶段高 ≥ 0.03，且 per-class F1 中 DISPUTED 不再是 0。

完成 Phase C 后：
```bash
python eval.py --predictions test-claims-predictions.json --groundtruth data/dev-claims.json
```
（用 dev 当对照只是格式验证，test 真值要等课程公布）—— 主要看不报错、152/153 条都被读到、标签合法。

---

## 风险与对策（精简版）

| 风险 | 应对 |
|---|---|
| FEVER 预训练算泄漏 | B1 必须先确认；不行则退 roberta-large-mnli（约掉 1-2 个点） |
| DeBERTa-v3-large T4 OOM | 退到 base 版本，损失 1-2% |
| Colab 12h 断连 | 每 epoch 存 ckpt 到 Drive |
| DISPUTED F1 仍为 0 | 把 class weight 提到 8-10x；focal loss γ=2 |
| 时间紧张 | Phase B 可以只做到 B3 stage 1+2，跳 B4 部分消融；保 C1-C2 不动 |
