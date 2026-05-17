"""Refresh three stale markdown cells (A3 header, A4.3 header, A4.5 header)
that still mention deleted A4.4 / hybrid retrieval / single ckpt path."""
import json

NB = 'Group_051__COMP90042_Project_2026.ipynb'

CELL_35_A3_HEADER = '''## A3: 三路 backbone 对照 — base / large / nli + 两阶段微调

**输入**：`[CLS] claim [SEP] evi_1 [SEP] evi_2 ... [SEP]`，attention-mask weighted mean-pooling 后接新训的 4 类分类头（不复用 backbone 原 NLI head）。

**三条路径**（在 cell A3.1 通过 `CLS_BACKBONE_CHOICE` 切换）：

| 路径 | Backbone | LoRA | 检索 | ckpt |
|---|---|---|---|---|
| `'base'` | `microsoft/deberta-v3-base`（180M） | 否（全量微调） | BM25 top-5 | `cls_best_base.pt` |
| `'large'` | `microsoft/deberta-v3-large`（435M） | r=16, α=32 | custom BGE+rerank top-5 | `cls_best_large.pt` |
| `'nli'`（最终） | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`（435M） | r=16, α=32 | custom BGE+rerank top-5 | `cls_best_nli.pt` |

NLI backbone 已在 MNLI / FEVER / ANLI / LingNLI / WANLI 上预微调，对 fact-checking 起点最高（A6 对照表会量化这一点）。
LoRA 目标模块 `query_proj` / `value_proj`，仅 LoRA delta + 分类头可训练，Colab T4 安全。

**两阶段训练**（每条路径都跑）：
1. **Stage 1 — gold supervision**：训练集 gold evidence，热身 backbone + 分类头。
2. **Stage 2 — retrieved supervision**：从 Stage 1 加载，换用 `final_train_evidence`（实验路径决定的检索 top-K）继续训练，消除 train-eval 检索分布不一致。

损失：class-weighted CE（inverse frequency, sum=4，对 DISPUTED 加权）+ label_smoothing=0.1；
优化器 AdamW + cosine warmup；fp16 (cuda only)。Best ckpt 按 dev_acc 选。'''


CELL_43_A4_3_HEADER = '''## A4.3: NLI 风格 per-evidence 推理（仅 `'nli'` 路径启用）

把 NLI-pretrained 的 backbone 当作「逐条 evidence 推断」的 NLI 模型用，作为 A4.1 拼接推理的**互补路径**：

- 对每条 evidence i 单独跑 `(claim, evi_i)` → 4 类 softmax `p_i`
- 默认输出 `argmax(mean_i p_i)`
- 优势：SUPPORTS / REFUTES 上比 A4.1 拼接强很多
- 短板：单条 evidence 推理时模型几乎从不输出 NEI / DISPUTED，整体 acc 反而低于 A4.1

> 原计划用「某条 evi 强支持 + 某条强反对 → DISPUTED」的 τ 规则触发 DISPUTED，
> 但 dev 上 τ-sweep 表明任何阈值都无法可靠触发（单 evi 概率分布偏 SUP/REF），
> 最终 τ=1.01（关闭规则）。DISPUTED 的修复留给训练侧（过采样 / focal loss）。

`base` / `large` 路径会自动跳过本 cell；预测写入 `nli-per-evi-*-predictions.json`，
由 **A4.5 hybrid routing** 与 A4.1 的 `nli-*-predictions.json` 融合后产出最终输出。'''


CELL_45_A4_5_HEADER = '''## A4.5: Hybrid routing —— NLI 路径最终采用方案

A4.1 拼接式 NLI 与 A4.3 per-evi NLI 在不同标签类别上**互补**：A4.1 在 NEI 上较保守可靠，A4.3 在 SUP/REF 上更准。基于 dev confusion matrix（A4.5 cell 跑完自动打印）观察到的分布，按如下规则路由：

```
if concat_pred (A4.1) == 'NOT_ENOUGH_INFO':
    final = NEI            # 信任 A4.1 拼接的 NEI 判断
else:
    final = per_evi_pred   # 在 SUPPORTS / REFUTES 之间用 A4.3
```

`base` / `large` 路径会自动跳过本 cell；NLI 路径下产出 `hybrid-{dev,test}-predictions.json` —— **作为最终主线提交文件**。

> 历史数字（一次实验结果，仅供参考；最终数字以 A6 cell 输出为准）：
> | 路径 | A | F | Hmean |
> |---|---|---|---|
> | A4.1 拼接（NLI） | 0.5260 | 0.1057 | 0.1761 |
> | A4.3 NLI 聚合 | 0.4675 | 0.1057 | 0.1725 |
> | **A4.5 Hybrid** | **0.5325** | **0.1057** | **0.1764** |'''


def replace(cells, cell_id, new_src):
    for c in cells:
        if c.get('id') == cell_id:
            c['source'] = new_src.splitlines(keepends=True)
            if c['source'] and c['source'][-1].endswith('\n'):
                c['source'][-1] = c['source'][-1].rstrip('\n')
            return
    raise SystemExit(f'cell {cell_id} not found')


def main():
    with open(NB) as f:
        nb = json.load(f)
    replace(nb['cells'], '430ec164', CELL_35_A3_HEADER)
    replace(nb['cells'], '711f85a7', CELL_43_A4_3_HEADER)
    replace(nb['cells'], '371ae36a', CELL_45_A4_5_HEADER)
    with open(NB, 'w') as f:
        json.dump(nb, f, indent=2)
    print('markdown cleanup applied')


if __name__ == '__main__':
    main()
