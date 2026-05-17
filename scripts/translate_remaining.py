"""Translate the remaining Chinese fragments (zero code semantics changed)."""
import json

NB = 'Group_051__COMP90042_Project_2026.ipynb'

# Line-level: exact-string replacements, applied across all cells.
REPLACEMENTS = [
    ('# 3) 本地：相对当前工作目录（在 notebook 同级目录启动 jupyter 即可）',
     '# 3) Local: relative to the current working directory (start jupyter from the notebook folder)'),
    ('# 设备选择：cuda > mps > cpu',
     '# Device priority: cuda > mps > cpu'),

    ("raise RuntimeError(f'{module_name} 未安装成功；请重新运行本 cell，必要时 Runtime > Restart runtime 后再 Run all')",
     "raise RuntimeError(f'{module_name} failed to install; rerun this cell, and if needed Runtime > Restart runtime followed by Run all')"),
    ("raise RuntimeError(f'下载 NLTK 资源 {pkg} 失败')",
     "raise RuntimeError(f'Failed to download NLTK resource {pkg}')"),
    ('# 新版 nltk 才有',
     '# only available on newer nltk versions'),
    ("print('✓ tokenizer 依赖就绪（sentencepiece / transformers / tokenizers）')",
     "print('Tokenizer dependencies ready (sentencepiece / transformers / tokenizers)')"),
    ("print('✓ NLTK 资源就绪')",
     "print('NLTK resources ready')"),

    ('# evidence.json 174MB，首次加载后写入 pickle；后续只读 pickle。',
     '# evidence.json is 174MB; we pickle it on first load and only read the pickle afterwards.'),
    ("print('从 pickle 缓存加载 evidence ...')",
     "print('Loading evidence from the pickle cache ...')"),
    ("print('首次加载 evidence.json (174MB) ...')",
     "print('First-time load of evidence.json (174MB) ...')"),
    ("print(f'\\n首 3 个 evi_id: {evi_ids[:3]}')",
     "print(f'\\nFirst 3 evidence ids: {evi_ids[:3]}')"),
    ("print(f'Evidence corpus 大小: {len(evi_ids):,} passages')",
     "print(f'Evidence corpus size: {len(evi_ids):,} passages')"),

    ('# Test 集对比（无 label / 无 gold evidences）',
     '# Test split comparison (no label, no gold evidences)'),
    ("print('Test 字段（应无 claim_label / evidences）:')",
     "print('Test fields (should not contain claim_label / evidences):')"),

    ('| 4 类标签分布 | 论证 class weight 必要性，DISPUTED 极少 |',
     '| 4-way label distribution | Motivates class weighting; DISPUTED is very rare |'),
    ('| 每条 claim 的 gold evidence 数 | 决定最终输出 evidence 的 Top-K 大小 |',
     '| Gold evidence count per claim | Informs the final output top-k size |'),
    ('| Evidence 长度（log scale） | 决定 evidence 端截断策略 |',
     '| Evidence length (log scale) | Informs the evidence-side truncation strategy |'),

    ('# Train claim 维度',
     '# Train-claim view'),
    ('# Evidence 长度（先算一遍，下面要用）',
     '# Evidence length (computed once; used by the panels below)'),
    ('# (1) claim 长度',
     '# (1) Claim length'),
    ('# (2) 4 类标签分布',
     '# (2) 4-way label distribution'),
    ('# (3) 每条 claim 的 gold evidence 数',
     '# (3) Gold evidence count per claim'),
    ('# (4) Evidence 长度（log y 因为长尾很重）',
     '# (4) Evidence length (log y because of the heavy tail)'),
    ('# 关键数字打印（写报告用）',
     '# Print headline numbers for the report'),
    ("print(f'Claim 长度: mean={df[\"claim_len\"].mean():.1f}, p95={df[\"claim_len\"].quantile(0.95):.0f}')",
     "print(f'Claim length: mean={df[\"claim_len\"].mean():.1f}, p95={df[\"claim_len\"].quantile(0.95):.0f}')"),
    ("print(f'Gold evidence 数: mean={df[\"n_evi\"].mean():.2f}, p95={df[\"n_evi\"].quantile(0.95):.0f}')",
     "print(f'Gold evidence count: mean={df[\"n_evi\"].mean():.2f}, p95={df[\"n_evi\"].quantile(0.95):.0f}')"),
    ("print(f'Evidence 长度: p50={int(np.percentile(evi_lens,50))}, '",
     "print(f'Evidence length: p50={int(np.percentile(evi_lens,50))}, '"),
    ("f'>256w 占比 {(evi_lens > 256).mean()*100:.2f}%')",
     "f'fraction >256 words: {(evi_lens > 256).mean()*100:.2f}%')"),

    ('# 后续只需要 filtered corpus；释放 raw evidence 降低 Colab RAM 峰值。',
     '# Only the filtered corpus is needed downstream; freeing raw evidence reduces Colab RAM peak.'),
]


def main():
    with open(NB) as f:
        nb = json.load(f)
    miss = 0
    for c in nb['cells']:
        src = ''.join(c['source'])
        new = src
        for old, new_txt in REPLACEMENTS:
            new = new.replace(old, new_txt)
        if new != src:
            c['source'] = new.splitlines(keepends=True)
            if c['source'] and c['source'][-1].endswith('\n'):
                c['source'][-1] = c['source'][-1].rstrip('\n')
    with open(NB, 'w') as f:
        json.dump(nb, f, indent=2)
    print(f'applied {len(REPLACEMENTS)} replacement patterns')


if __name__ == '__main__':
    main()
