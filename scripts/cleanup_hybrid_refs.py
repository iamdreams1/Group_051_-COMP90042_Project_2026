"""Strip stale hybrid references from cell 5 (file existence check) and cell 20
(dense-dependency lazy install). Both still mention hybrid_*.pkl after the main
integration edit; without this fix cell 20 KeyErrors on FORCE_REBUILD['hybrid']."""
import json

NB = 'Group_051__COMP90042_Project_2026.ipynb'

CELL_5_NEW = '''# 1.1.b 数据就位检查（团队共享 cache：避免每人重新跑重型步骤）
#
# 这一步只做"文件是否就位"的快速检查，不加载大对象。
# 真正加载/校验由后面的 cache helper 统一处理。
import os

REQUIRED = {
    'data/train-claims.json'           : True,
    'data/dev-claims.json'             : True,
    'data/test-claims-unlabelled.json' : True,
    'data/evidence.json'               : True,
}

OFFICIAL_CACHE = {
    'cache/evidence.pkl'          : 'raw evidence pickle',
    'cache/evidence_filtered.pkl' : 'filtered evidence dict + ids',
    'cache/bm25.pkl'              : 'BM25Okapi index',
    'cache/bge_evidence_emb.npy'  : 'raw BGE evidence embeddings (A1.a ablation)',
    'cache/bge_faiss.index'       : 'raw BGE FAISS index (A1.a ablation)',
    'cache/retrieval_eval_dev.pkl': 'A1.a retrieval Recall@K table',
    'cache/bm25_dev.pkl'          : 'bm25 dev candidates (baseline / Exp-1)',
    'cache/bm25_train.pkl'        : 'bm25 train candidates (baseline / Exp-1)',
    'cache/bm25_test.pkl'         : 'bm25 test candidates (baseline / Exp-1)',
    # Custom BGE retriever cache (filled by A2.b on first run, ~3.5 GB)
    'cache/custom_bge_evidence_emb.npy': 'fine-tuned BGE evidence embeddings',
    'cache/custom_bge_faiss.index'    : 'fine-tuned BGE FAISS index',
    'cache/custom_bge_evi_ids.pkl'    : 'fine-tuned BGE evidence id order',
}

print(f'检查 ROOT = {ROOT}')
print('-' * 70)
missing_required = []
for rel, required in REQUIRED.items():
    p = os.path.join(ROOT, rel)
    if os.path.exists(p):
        print(f'  ok {rel:<42s} {os.path.getsize(p)/1e6:>8.1f} MB')
    else:
        print(f'  MISSING {rel}')
        if required:
            missing_required.append(rel)

if missing_required:
    raise FileNotFoundError(
        f'缺少必需文件: {missing_required}\\n'
        f'请确认 Drive shortcut 已生效，或在本地把 data/ 放到 {ROOT} 下。'
    )
print('-' * 70)
print('必需数据已就位')

print('\\n官方 cache 快速检查（存在不代表有效，后面会逐项校验）')
print('-' * 70)
missing_cache = []
for rel, desc in OFFICIAL_CACHE.items():
    p = os.path.join(ROOT, rel)
    if os.path.exists(p):
        print(f'  ok {rel:<34s} {os.path.getsize(p)/1e6:>8.1f} MB  ({desc})')
    else:
        print(f'  missing {rel:<34s} ({desc})')
        missing_cache.append(rel)

# Fast-path summary：对应三条实验路径的 cache 就位情况
BM25_CACHE_RELS = ['cache/bm25_dev.pkl', 'cache/bm25_train.pkl', 'cache/bm25_test.pkl']
CUSTOM_BGE_RELS = ['cache/custom_bge_evidence_emb.npy', 'cache/custom_bge_faiss.index', 'cache/custom_bge_evi_ids.pkl']
ALL_BM25_FILES_PRESENT       = all(os.path.exists(os.path.join(ROOT, rel)) for rel in BM25_CACHE_RELS)
ALL_CUSTOM_BGE_FILES_PRESENT = all(os.path.exists(os.path.join(ROOT, rel)) for rel in CUSTOM_BGE_RELS)
print('\\nFast-path summary:')
print(f'  Exp-1 baseline (BM25) cache present       : {ALL_BM25_FILES_PRESENT}')
print(f'  Exp-2/3 custom BGE corpus cache present   : {ALL_CUSTOM_BGE_FILES_PRESENT}')
print(f'                   (若 False，cell A2.b 首次运行会重建，约 30 分钟)')
'''


CELL_20_NEW = '''# 2.2 Dense retrieval dependencies（仅在需要 BGE/FAISS 时安装 faiss）
import importlib.util, subprocess, sys, os

EVAL_CACHE_PATH = f'{CACHE}/retrieval_eval_dev.pkl'

NEED_DENSE_DEPS = (
    FORCE_REBUILD['bge'] or
    FORCE_REBUILD['custom_bge'] or
    FORCE_REBUILD['eval'] or
    (not os.path.exists(EVAL_CACHE_PATH)) or
    (not os.path.exists(f'{CACHE}/custom_bge_evidence_emb.npy'))
)

if NEED_DENSE_DEPS and importlib.util.find_spec('faiss') is None:
    print('Installing faiss-cpu because dense retrieval may be needed...')
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'faiss-cpu'])
else:
    print(f'Dense dependency check complete. NEED_DENSE_DEPS={NEED_DENSE_DEPS}')
'''


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
    replace(nb['cells'], 'e0c87bda', CELL_5_NEW)
    replace(nb['cells'], '6deed960', CELL_20_NEW)
    with open(NB, 'w') as f:
        json.dump(nb, f, indent=2)
    print('cleanup applied')


if __name__ == '__main__':
    main()
