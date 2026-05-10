# COMP90042 Group_051 — 可执行实施计划

> 这是一份"操作手册"。每一步给出**做什么 / 怎么做 / 验收标准 / 命令或代码片段**。
> 按顺序执行；每完成一步就在前面方框里打 `[x]`。

---

## 0. 总览

- **任务**：气候 claim 自动事实核查 = 证据检索 (F) + 4 分类 (A)，主指标 `Hmean = 2FA/(F+A)`
- **架构**：**Hybrid (BM25 ∪ BGE-small Dense) 召回 ≤400** → Cross-Encoder 重排 Top-K(≤6) → DeBERTa-v3-base + gated-attention 4 分类
- **截止**：2026-05-22（剩 18 天）
- **运行环境**：Google Colab 免费版（T4 16GB / 12GB RAM）
- **唯一交付代码文件**：`Group_051__COMP90042_Project_2026.ipynb`
- **数据规模**（已校验）：train=1228 / dev=154 / test=153 / evidence=1.2M 段（174MB）
- **关键约束**：
  - 必须含 RNN/LSTM/GRU/Transformer 序列模型
  - 仅开源模型（禁 GPT/Claude/Gemini API）
  - evidence 预测 ≤6 条（eval.py 第 44 行 `top_six_ev` 暗示）

---

## 1. 环境准备（5/4，30 分钟）

### Step 1.1 — Colab + Drive 挂载

- [ ] 在 Colab 新建 notebook（或上传本仓 ipynb），运行：

```python
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/comp90042/{ckpt,cache}
```

- [ ] 把整个项目目录上传到 `/content/drive/MyDrive/comp90042/`

### Step 1.2 — 安装依赖

- [ ] 在 notebook 第一个 cell：

```python
!pip install -q rank_bm25 sentence-transformers==2.7.0 transformers==4.40.0 \
                 nltk pandas matplotlib scikit-learn
import nltk; nltk.download('punkt'); nltk.download('stopwords')
```

**验收**：`import` 全部不报错，`torch.cuda.is_available() == True`。

---

## 2. M1 — 数据加载与 EDA（5/4–5/5，写入"1. DataSet Processing" section）

### Step 2.1 — 读取四个 json + evidence 缓存

- [ ] 写一个 cell：

```python
import json, pickle, os
DATA = '/content/drive/MyDrive/comp90042/data'
CACHE = '/content/drive/MyDrive/comp90042/cache'

def load_json(name):
    return json.load(open(f'{DATA}/{name}.json'))

train = load_json('train-claims')         # 1228
dev   = load_json('dev-claims')           # 154
test  = load_json('test-claims-unlabelled')  # 153
baseline = load_json('dev-claims-baseline')  # 154

# evidence 大文件 → pickle 缓存
evi_pkl = f'{CACHE}/evidence.pkl'
if os.path.exists(evi_pkl):
    evidence = pickle.load(open(evi_pkl, 'rb'))
else:
    evidence = json.load(open(f'{DATA}/evidence.json'))
    pickle.dump(evidence, open(evi_pkl, 'wb'))

assert len(train)==1228 and len(dev)==154 and len(test)==153
print(f'evidence: {len(evidence):,} 条')
```

**验收**：assert 通过；evidence ≥ 1,200,000 条。

### Step 2.2 — EDA（必须出图，报告里要用）

- [ ] 写一个 cell 出三张图：
  1. **claim 长度分布**（word count histogram）
  2. **4 类标签分布**（bar chart，预期 DISPUTED 极少）
  3. **每条 claim 的 gold evidence 数分布**（bar chart，预期 1–5 之间，均值 ≈3）

```python
import pandas as pd, matplotlib.pyplot as plt
from collections import Counter

df = pd.DataFrame([{
    'claim_len': len(v['claim_text'].split()),
    'label': v['claim_label'],
    'n_evi': len(v['evidences'])
} for v in train.values()])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
df['claim_len'].hist(ax=axes[0], bins=30); axes[0].set_title('Claim length')
df['label'].value_counts().plot.bar(ax=axes[1]); axes[1].set_title('Label dist')
df['n_evi'].value_counts().sort_index().plot.bar(ax=axes[2]); axes[2].set_title('# gold evi')
plt.tight_layout(); plt.show()

print(df['label'].value_counts(normalize=True))
```

**验收**：三张图正常显示；DISPUTED 占比 < 10%（说明类不均衡，需 class weight）。

---

## 3. M2-a — Hybrid 检索（已完成，写入"2. Model Implementation"）

> ✅ **已完成（截至 5/4）**：原计划只有 BM25，实测 Recall@200=0.494 远低于 0.75 验收线。
> 改为 BM25 ∪ BGE-small dense 并集，候选 ≤400，Recall@400 = **0.844**。
>
> | K | BM25 | Dense | Hybrid |
> |---|------|-------|--------|
> | 50 | 0.353 | 0.474 | 0.577 |
> | 200 | 0.494 | 0.685 | 0.769 |
> | 400 | 0.569 | 0.771 | **0.844** |
>
> 这张表本身就是报告里的一个核心消融实验（创新点 ① — Hybrid 检索的有效性）。
> `cache/hybrid_{train,dev,test}.pkl` 候选已缓存，下游直接复用。

### （历史记录）M2-a 旧 BM25-only baseline

### Step 3.1 — 分词与索引

- [ ] cell：构建 BM25 索引并 pickle 缓存（首次约 5 分钟，之后 30 秒）

```python
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
STOP = set(stopwords.words('english'))

def tok(text):
    return [w.lower() for w in word_tokenize(text) if w.isalpha() and w.lower() not in STOP]

idx_pkl = f'{CACHE}/bm25.pkl'
if os.path.exists(idx_pkl):
    bm25, evi_ids = pickle.load(open(idx_pkl, 'rb'))
else:
    evi_ids = list(evidence.keys())
    corpus = [tok(evidence[i]) for i in evi_ids]   # 1.2M 段，约 5 min
    bm25 = BM25Okapi(corpus)
    pickle.dump((bm25, evi_ids), open(idx_pkl, 'wb'))
```

### Step 3.2 — 检索 + Recall 评估

- [ ] cell：dev 上跑 Recall@{50,100,200}

```python
import numpy as np
def retrieve(claim_text, k=200):
    scores = bm25.get_scores(tok(claim_text))
    top = np.argpartition(-scores, k)[:k]
    top = top[np.argsort(-scores[top])]
    return [evi_ids[i] for i in top]

def recall_at_k(k):
    rs = []
    for cid, c in dev.items():
        top = set(retrieve(c['claim_text'], k))
        gold = set(c['evidences'])
        rs.append(len(top & gold) / len(gold))
    return np.mean(rs)

for k in [50, 100, 200]:
    print(f'Recall@{k} = {recall_at_k(k):.3f}')
```

**验收**：`Recall@200 ≥ 0.75`；不到则检查分词是否过滤过狠。

### Step 3.3 — 缓存 dev/train BM25 Top-200

- [ ] 推理一次后存 pickle（reranker 训练 + 推理都要复用）：

```python
def cache_topk(claims, k=200, name=''):
    out = {cid: retrieve(c['claim_text'], k) for cid, c in claims.items()}
    pickle.dump(out, open(f'{CACHE}/bm25_top{k}_{name}.pkl', 'wb'))
    return out

bm25_train = cache_topk(train, 200, 'train')
bm25_dev   = cache_topk(dev,   200, 'dev')
bm25_test  = cache_topk(test,  200, 'test')
```

**验收**：三个 pickle 文件生成；每条 claim 对应 200 个 evidence id。

---

## 4. M2-b — Cross-Encoder Reranker（5/5–5/8，notebook cell 23-27 已写）

> 输入从 BM25 Top-200 改成 **Hybrid ≤400**，hard neg 也来自 hybrid 候选（不是 BM25）。
> 训练对总数变化不大（gold + 4 hard neg ≈ 8.6K 对），但负样本难度更高，reranker 学到的判别边界更准。

### Step 4.1 — 构造训练对

- [ ] cell：每条 train claim → 所有 gold (label=1) + hybrid 候选中前 4 个非 gold (label=0)

```python
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
import random; random.seed(42)

pairs = []
for cid, c in train.items():
    gold = set(c['evidences'])
    for g in gold:
        pairs.append(InputExample(texts=[c['claim_text'], evidence[g]], label=1.0))
    negs = [e for e in bm25_train[cid] if e not in gold][:4]
    for n in negs:
        pairs.append(InputExample(texts=[c['claim_text'], evidence[n]], label=0.0))
print(len(pairs), '训练对')   # ≈ 1228*(3+4) ≈ 8.6K
```

### Step 4.2 — 微调（2 epoch，T4 约 15 min）

- [ ] cell：

```python
model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', num_labels=1, max_length=256)
loader = DataLoader(pairs, batch_size=16, shuffle=True)
model.fit(train_dataloader=loader, epochs=2, warmup_steps=100,
          output_path=f'/content/drive/MyDrive/comp90042/ckpt/reranker')
```

### Step 4.3 — 重排 + K 调优

- [ ] cell：dev 上对 BM25 Top-200 重排，选最优 K

```python
def rerank(claims, bm25_topk):
    out = {}
    for cid, c in claims.items():
        cands = bm25_topk[cid]
        scores = model.predict([(c['claim_text'], evidence[e]) for e in cands],
                               batch_size=64, show_progress_bar=False)
        order = np.argsort(-scores)
        out[cid] = [cands[i] for i in order]
    return out

reranked_dev = rerank(dev, bm25_dev)

def f_at_k(reranked, claims, k):
    fs = []
    for cid, c in claims.items():
        pred = set(reranked[cid][:k]); gold = set(c['evidences'])
        tp = len(pred & gold)
        if tp == 0: fs.append(0.0); continue
        p, r = tp/k, tp/len(gold)
        fs.append(2*p*r/(p+r))
    return np.mean(fs)

for k in [3,4,5,6]:
    print(f'F@{k} = {f_at_k(reranked_dev, dev, k):.3f}')
```

**验收**：至少一个 K 上 `F ≥ 0.20`（baseline F ≈ 0.10）。记下最优 K（多半是 4 或 5）。

### Step 4.4 — 缓存重排结果

- [ ] 同样跑 train / test 并 pickle，供分类阶段用。

---

## 5. M3 — DeBERTa 4 分类（5/12–5/14，写入"2. Model Implementation"后半）

### Step 5.1 — 数据集类

- [ ] cell：拼 `[CLS] claim [SEP] evi_1 [SEP] evi_2 [SEP] ... [SEP]`，max_len=512

```python
from transformers import AutoTokenizer
import torch

LABELS = ['SUPPORTS','REFUTES','NOT_ENOUGH_INFO','DISPUTED']
L2I = {l:i for i,l in enumerate(LABELS)}
tok_cls = AutoTokenizer.from_pretrained('microsoft/deberta-v3-base')

class ClaimDS(torch.utils.data.Dataset):
    def __init__(self, claims, evi_dict, use_gold=True, top_k=5):
        self.items = []
        for cid, c in claims.items():
            evis = c['evidences'] if use_gold else evi_dict[cid][:top_k]
            text_evi = ' [SEP] '.join(evidence[e] for e in evis)
            enc = tok_cls(c['claim_text'], text_evi, truncation=True,
                          max_length=512, padding='max_length', return_tensors='pt')
            self.items.append((enc, L2I.get(c.get('claim_label','NOT_ENOUGH_INFO'), 2)))
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        enc, y = self.items[i]
        return {k: v.squeeze(0) for k,v in enc.items()}, torch.tensor(y)
```

### Step 5.2 — 模型 + 训练（OOP，写入"OOP代码区"）

- [ ] cell：自写 `Trainer` 类（fp16 + class weight + early stopping）

```python
from transformers import AutoModel, get_cosine_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
import torch.nn as nn

class GatedClassifier(nn.Module):
    """创新点：每条 evidence 单独编码，用 gated attention 聚合"""
    def __init__(self, base='microsoft/deberta-v3-base', n=4):
        super().__init__()
        self.enc = AutoModel.from_pretrained(base)
        h = self.enc.config.hidden_size
        self.gate = nn.Linear(h, 1)
        self.cls  = nn.Linear(h, n)
    def forward(self, ids, mask):
        h = self.enc(ids, attention_mask=mask).last_hidden_state[:,0]   # [B, H]
        # 单证据版（先跑通），多证据 gated 聚合留作创新点 v2
        return self.cls(h)

class Trainer:
    def __init__(self, model, train_ds, dev_ds, weights, lr=2e-5, epochs=4, bs=8):
        self.m = model.cuda()
        self.tl = torch.utils.data.DataLoader(train_ds, batch_size=bs, shuffle=True)
        self.dl = torch.utils.data.DataLoader(dev_ds, batch_size=bs)
        self.opt = torch.optim.AdamW(model.parameters(), lr=lr)
        self.sch = get_cosine_schedule_with_warmup(self.opt, len(self.tl)//10, len(self.tl)*epochs)
        self.crit = nn.CrossEntropyLoss(weight=torch.tensor(weights).float().cuda(),
                                        label_smoothing=0.1)
        self.scaler = GradScaler()
        self.epochs = epochs

    def run(self):
        best = 0
        for ep in range(self.epochs):
            self.m.train()
            for batch, y in self.tl:
                batch = {k:v.cuda() for k,v in batch.items()}; y = y.cuda()
                self.opt.zero_grad()
                with autocast():
                    logits = self.m(batch['input_ids'], batch['attention_mask'])
                    loss = self.crit(logits, y)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.m.parameters(), 1.0)
                self.scaler.step(self.opt); self.scaler.update(); self.sch.step()
            acc = self.evaluate()
            print(f'epoch {ep} dev_acc={acc:.3f}')
            if acc > best:
                best = acc
                torch.save(self.m.state_dict(), '/content/drive/MyDrive/comp90042/ckpt/cls_best.pt')
        return best

    @torch.no_grad()
    def evaluate(self):
        self.m.eval(); correct = total = 0
        for batch, y in self.dl:
            batch = {k:v.cuda() for k,v in batch.items()}; y = y.cuda()
            pred = self.m(batch['input_ids'], batch['attention_mask']).argmax(-1)
            correct += (pred == y).sum().item(); total += y.size(0)
        return correct / total
```

### Step 5.3 — 训练

- [ ] cell：用 gold evidence 训练 3 epoch，最后 1 epoch 用 reranker Top-K 微调

```python
# 类权重：1/freq 归一化
from collections import Counter
cnt = Counter(v['claim_label'] for v in train.values())
weights = [1/cnt[l] for l in LABELS]
w_sum = sum(weights); weights = [w/w_sum*4 for w in weights]

train_gold = ClaimDS(train, None, use_gold=True)
dev_gold   = ClaimDS(dev,   None, use_gold=True)

model = GatedClassifier()
trainer = Trainer(model, train_gold, dev_gold, weights, epochs=3, bs=8)
trainer.run()
```

**验收**：dev_acc ≥ 0.45（随机 0.25）。如果 OOM：`bs=4 + grad_accum=2` 或换 `roberta-base`。

---

## 6. M4 — 端到端 + dev 评估（5/15–5/16）

### Step 6.1 — 拼接流水线

- [ ] cell：写 `predict(claims) → {cid: {claim_text, claim_label, evidences}}`

```python
def predict(claims, reranked, k=5):
    out = {}
    model.eval()
    for cid, c in claims.items():
        evis = reranked[cid][:k]
        text_evi = ' [SEP] '.join(evidence[e] for e in evis)
        enc = tok_cls(c['claim_text'], text_evi, truncation=True,
                      max_length=512, padding='max_length', return_tensors='pt')
        with torch.no_grad(), autocast():
            ids = enc['input_ids'].cuda(); m = enc['attention_mask'].cuda()
            pred = model(ids, m).argmax(-1).item()
        out[cid] = {'claim_text': c['claim_text'],
                    'claim_label': LABELS[pred],
                    'evidences': evis}
    return out

dev_pred = predict(dev, reranked_dev, k=5)  # 用 Step 4.3 选的最优 K
json.dump(dev_pred, open('dev-predictions.json','w'))
```

### Step 6.2 — 调用官方 eval.py

- [ ] terminal：

```bash
!python eval.py --predictions dev-predictions.json \
                --groundtruth data/dev-claims.json
```

**验收**：`Hmean ≥ 0.15`（baseline ≈ 0.06）。

---

## 7. M5 — 创新点强化（5/17–5/18，拿 19 分方法分）

> **已实现的创新点**：
> ① **Hybrid retrieval (BM25 ∪ Dense)**：解决纯 BM25 的词汇鸿沟（"global warming" vs "climate change"），Recall@400 从 0.569 → 0.844。报告里直接用上面那张消融表。
>
> **下面三个是可选叠加**，按优先级 7.2 > 7.1 > 7.3。时间紧只做 7.2 即可。

### Step 7.1 — Reranker contrastive loss

- [ ] 把 Step 4.2 的 BCE loss 换成 InfoNCE：每条 claim 在 batch 内对比 1 个 gold vs N 个 hard neg
- [ ] 重新训练，对比 F-score 的提升

### Step 7.2 — Multi-evidence gated attention 聚合

- [ ] 改 `GatedClassifier.forward`：对 K 条 evidence 分别 encode 取 [CLS]，过 `gate(h_i)` softmax，加权求和后送 cls 头
- [ ] 报告里画"单 evi 版 vs gated 版"消融对比

### Step 7.3 — DISPUTED 数据增强（可选）

- [ ] 用 `nlpaug` back-translation 把 DISPUTED 样本扩 2 倍

---

## 8. 提交准备（5/19–5/22）

### Step 8.1 — Test 集推理

- [ ] cell：

```python
test_pred = predict(test, reranked_test, k=5)
json.dump(test_pred, open('test-claims-predictions.json','w'))
# 校验
assert len(test_pred) == 153
assert all('claim_label' in v and 'evidences' in v and 1 <= len(v['evidences']) <= 6
           for v in test_pred.values())
```

### Step 8.2 — Notebook 整理

- [ ] 完整重跑一遍 ipynb，**保留所有 cell 输出**（评估时会被检查）
- [ ] 把每个 section 顶部加 markdown 说明（方法、为什么这样选）
- [ ] 删掉所有调试用的 print

### Step 8.3 — 报告（≤7 页 ACL）

按以下结构写：
1. **Abstract**（150 词）
2. **Introduction**：任务定义 + 主要贡献（contrastive reranker, gated agg）
3. **Method**：架构图（必画） + 公式
4. **Experiments**：
   - **检索消融表**（必备）：BM25-only / Dense-only / **Hybrid** / +reranker，各行 Recall@K + F@K
   - **分类消融表**：单 evi / 多 evi 拼接 / **gated attention 聚合**，各行 A + per-class F
   - **端到端表**：与 `dev-claims-baseline.json` 对比 F / A / Hmean
5. **Error Analysis**：抽 10 条 DISPUTED 误分类做定性分析
6. **Conclusion**
7. **Team Contribution**

### Step 8.4 — 打包提交

- [ ] ZIP 包含：
  - `Group_051__COMP90042_Project_2026.ipynb`（含输出）
  - `test-claims-predictions.json`
  - `README.md`
  - 不要包含 `data/` 和 ckpt 文件
- [ ] PDF 报告单独提交

---

## 9. 风险清单 & 应对

| 风险 | 应对 |
|---|---|
| Colab 12h 断连 | 每 epoch 存 ckpt 到 Drive，下次 `model.load_state_dict()` |
| OOM | `gradient_checkpointing_enable()` + bs=4 + grad_accum=2；最坏退 `roberta-base` |
| ~~BM25 Recall@200 < 0.75~~ | ~~检查分词；考虑 claim 关键短语扩展~~ → **已通过 hybrid 解决**（Recall@400=0.844） |
| Hybrid 候选数过多导致 reranker 慢 | 把 K_EACH 从 200 降到 150（dev 候选 ≤300）；或 BGE 换 fp16 |
| BGE 编码全量 evidence OOM | encoder.max_seq_length=128（已设）；batch_size 调小到 32 |
| DISPUTED 永远预测不出 | 提高其 class weight 到 5–10x；back-translation 增强 |
| eval.py 报错 | 检查 test 输出 json 格式：每个 cid 必须有 `claim_text` `claim_label` `evidences` 三个 key |

---

## 10. 文件位置速查

| 用途 | 路径 |
|---|---|
| 主交付 notebook | `Group_051__COMP90042_Project_2026.ipynb` |
| 评估脚本（只读） | `eval.py` |
| 训练数据 | `data/train-claims.json` |
| 验证数据 | `data/dev-claims.json` |
| 测试数据 | `data/test-claims-unlabelled.json` |
| 证据库 | `data/evidence.json` |
| Baseline 对照 | `data/dev-claims-baseline.json` |
| Drive 缓存 | `/content/drive/MyDrive/comp90042/cache/` |
| Drive checkpoint | `/content/drive/MyDrive/comp90042/ckpt/` |
| 最终提交 | `test-claims-predictions.json` |
