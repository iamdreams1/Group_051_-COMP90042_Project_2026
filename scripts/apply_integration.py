"""Apply 3-path integration edits to the project notebook.

Idempotent: rerunning produces the same result.
Run from project root: python scripts/apply_integration.py
"""
import json
import os
import sys
import uuid

NB_PATH = 'Group_051__COMP90042_Project_2026.ipynb'


def make_id():
    return uuid.uuid4().hex[:12]


def find_cell(nb, cell_id):
    for i, c in enumerate(nb['cells']):
        if c.get('id') == cell_id:
            return i, c
    raise SystemExit(f'cell id {cell_id!r} not found')


def replace_source(cell, src):
    if isinstance(src, str):
        cell['source'] = src.splitlines(keepends=True)
        # ensure last line has no trailing newline (notebook convention)
        if cell['source'] and cell['source'][-1].endswith('\n'):
            cell['source'][-1] = cell['source'][-1].rstrip('\n')
    else:
        cell['source'] = list(src)


# ----------------------------------------------------------------------
# Cell sources (verbatim final state)
# ----------------------------------------------------------------------

CELL_7_CACHE_META = '''# 1.2.b Cache helpers（必须先于所有重型 cache 使用）
import os, pickle, gc, time

# 实验路径全景（三条线）：
#   Exp-1 baseline : BM25            → DeBERTa-v3-base
#   Exp-2 large    : custom BGE+rerank → DeBERTa-v3-large + LoRA
#   Exp-3 NLI      : custom BGE+rerank → MoritzLaurer NLI + LoRA   ← 最终采用
# 取消了 Hybrid BM25∪BGE 这条路径，避免和 custom_end_to_end 重复

FORCE_REBUILD = {
    'filtered': False,
    'bm25': False,
    'bge': False,
    'custom_bge': False,
    'eval': False,
}

CACHE_META = {
    'filtered': {
        'version': 1,
        'max_words': 256,
        'min_words_threshold': 3,
    },
    'bm25': {
        'version': 1,
        'tokenizer': 'nltk_word_tokenize_lower_stopwords_punct_isalnum_v1',
        'max_words': 256,
        'min_words_threshold': 3,
    },
    'bge': {
        # raw BGE — A1.a 检索消融对照用（vs custom 微调版）
        'version': 1,
        'model': 'BAAI/bge-base-en-v1.5',
        'max_seq_len': 384,
        'embedding_dim': 768,
    },
    'custom_bge': {
        # 微调后的 BGE retriever（Drive: ckpt/custom-bge-retriever-final）
        # + 微调后的 reranker（Drive: ckpt/custom-bge-reranker-final）
        # 主线（Exp-2 / Exp-3）使用这条
        'version': 1,
        'base_model': 'BAAI/bge-base-en-v1.5',
        'retriever_ckpt': 'ckpt/custom-bge-retriever-final',
        'reranker_ckpt': 'ckpt/custom-bge-reranker-final',
        'max_seq_len': 384,
        'embedding_dim': 768,
    },
}


def save_pickle_atomic(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def release_large_objects(*names):
    released = []
    for name in names:
        if name in globals():
            del globals()[name]
            released.append(name)
    if released:
        gc.collect()
        if 'torch' in globals() and getattr(torch, 'cuda', None) and torch.cuda.is_available():
            torch.cuda.empty_cache()
        print('released:', ', '.join(released))


def validate_candidate_cache(data, claims, name='cache', max_len=None, known_evidence=None):
    if not isinstance(data, dict):
        return False, f'{name} is not a dict'
    expected = set(claims.keys())
    actual = set(data.keys())
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        return False, f'{name} claim ids mismatch: missing={len(missing)} extra={len(extra)}'
    for cid, vals in data.items():
        if not isinstance(vals, list):
            return False, f'{name}[{cid}] is not a list'
        if max_len is not None and len(vals) > max_len:
            return False, f'{name}[{cid}] has {len(vals)} candidates > {max_len}'
        if len(vals) != len(set(vals)):
            return False, f'{name}[{cid}] contains duplicate evidence ids'
        if known_evidence is not None:
            bad = [eid for eid in vals if eid not in known_evidence]
            if bad:
                return False, f'{name}[{cid}] contains unknown evidence id {bad[0]}'
    return True, 'ok'


def load_candidate_cache(path, claims, name='cache', max_len=None, known_evidence=None, force=False):
    if force or not os.path.exists(path):
        reason = 'forced rebuild' if force else 'missing file'
        print(f'  {name}: {reason}')
        return None
    try:
        t0 = time.time()
        data = load_pickle(path)
        ok, msg = validate_candidate_cache(data, claims, name=name, max_len=max_len, known_evidence=known_evidence)
        if ok:
            avg = sum(len(v) for v in data.values()) / max(1, len(data))
            print(f'  loaded {path} ({len(data)} claims, avg {avg:.0f}, {time.time()-t0:.1f}s)')
            return data
        print(f'  invalid {name}: {msg}; rebuilding')
    except Exception as e:
        print(f'  failed to load {name} ({type(e).__name__}: {e}); rebuilding')
    return None

print('Cache-first helpers ready')
print('FORCE_REBUILD =', FORCE_REBUILD)
'''


CELL_23_A1_HEADER = '''## A1: 检索消融 — BM25 / Custom BGE / Custom BGE + Reranker 的 Recall@K

三条检索路径对照，对应「检索升级」实验叙事：

1. **BM25**（baseline 实验用，喂给 DeBERTa-v3-base）
2. **Custom BGE**（dense, 微调版）
3. **Custom BGE + Reranker**（最终方案，喂给 DeBERTa-v3-large / MoritzLaurer NLI）

下方 cell A1.a 计算三路 Recall@K，A1.c 缓存 BM25 候选 (k=200) 供 baseline 实验使用。
Custom BGE+Reranker 的候选由 cell A2.b 动态生成（不缓存，因为只算 dev/test 153×2 条，速度可控）。'''


CELL_24_A1_A_EVAL = '''# A1.a: 检索评估 — BM25 vs Custom BGE vs Custom BGE+Reranker（dev 154 条）
# 输出：表格 + cache/retrieval_eval_dev.pkl（idempotent）
import numpy as np
import time
from tqdm.auto import tqdm

EVAL_CACHE_PATH = f'{CACHE}/retrieval_eval_dev.pkl'

def recall_at_k(candidate_dict, claims, k):
    rs = []
    for cid, c in claims.items():
        pred = set(candidate_dict.get(cid, [])[:k])
        gold = set(c['evidences'])
        if not gold:
            continue
        rs.append(len(pred & gold) / len(gold))
    return float(np.mean(rs)) if rs else 0.0

def f_at_k_simple(candidate_dict, claims, k):
    fs = []
    for cid, c in claims.items():
        pred = set(candidate_dict.get(cid, [])[:k])
        gold = set(c['evidences'])
        if not pred or not gold:
            fs.append(0.0)
            continue
        tp = len(pred & gold)
        if tp == 0:
            fs.append(0.0)
            continue
        p, r = tp / len(pred), tp / len(gold)
        fs.append(2 * p * r / (p + r))
    return float(np.mean(fs)) if fs else 0.0


# A1.a 只算这一张表，不缓存所有 train candidates（避免和 A1.c / A2.b 重复存储）
KS = [3, 5, 10, 50, 100]
results = {}

# --- 1) BM25 单路（top-100 即可覆盖 K 范围） ---
print('[A1.a] BM25 top-100 ...')
t0 = time.time()
bm25_dev_cand = {cid: [d['id'] for d in get_top_200(c['claim_text'], n=100)]
                 for cid, c in tqdm(dev.items(), desc='BM25')}
print(f'  bm25 done in {time.time()-t0:.1f}s')
results['BM25'] = {k: recall_at_k(bm25_dev_cand, dev, k) for k in KS}

# --- 2) Custom BGE 单路（lazy-load 微调版 retriever） ---
print('\\n[A1.a] Custom BGE top-100 ...')
try:
    from sentence_transformers import SentenceTransformer
    import faiss as _faiss
    custom_retriever_path = f'{CKPT}/custom-bge-retriever-final'
    if not os.path.isdir(custom_retriever_path):
        raise FileNotFoundError(f'{custom_retriever_path} missing — 见 README "Drive checkpoint 同步"')
    _retr = SentenceTransformer(custom_retriever_path, device=DEVICE)
    _emb_path = f'{CACHE}/custom_bge_evidence_emb.npy'
    _idx_path = f'{CACHE}/custom_bge_faiss.index'
    _ids_path = f'{CACHE}/custom_bge_evi_ids.pkl'
    if os.path.exists(_emb_path) and os.path.exists(_idx_path) and os.path.exists(_ids_path):
        _idx = _faiss.read_index(_idx_path)
        _eids = load_pickle(_ids_path)
    else:
        print('  custom-bge corpus 编码缓存不存在，跳过 Custom BGE 评估（请先跑 cell A2.b 生成）')
        raise RuntimeError('skip custom_bge eval')
    bge_dev_cand = {}
    for cid, c in tqdm(dev.items(), desc='Custom BGE'):
        q = _retr.encode([c['claim_text']], normalize_embeddings=True).astype('float32')
        _, ix = _idx.search(q, 100)
        bge_dev_cand[cid] = [_eids[j] for j in ix[0]]
    results['Custom BGE'] = {k: recall_at_k(bge_dev_cand, dev, k) for k in KS}
    del _retr, _idx
except Exception as e:
    print(f'  Custom BGE evaluation skipped ({type(e).__name__}: {e})')

# --- 3) Custom BGE + Reranker（依赖 cell A2.b 的 final_dev_evidence；可选） ---
if 'final_dev_evidence' in globals() and final_dev_evidence is not None and RETRIEVAL_SOURCE == 'custom_end_to_end':
    print('\\n[A1.a] Custom BGE + Reranker (reusing final_dev_evidence from A2.b) ...')
    results['Custom BGE + Reranker'] = {k: recall_at_k(final_dev_evidence, dev, k)
                                          for k in KS if k <= FINAL_TOP_K + 5}
else:
    print('\\n[A1.a] Reranker 行需先跑 cell A2.b（RETRIEVAL_SOURCE=custom_end_to_end）')

# --- print table ---
print('\\n=== Recall@K (dev) ===')
header = f'{"Retriever":<26s}' + '  '.join(f'R@{k:>3d}' for k in KS)
print(header)
print('-' * len(header))
for name in ['BM25', 'Custom BGE', 'Custom BGE + Reranker']:
    if name not in results:
        continue
    row = f'{name:<26s}'
    for k in KS:
        v = results[name].get(k)
        row += f'  {v:>4.3f}' if v is not None else '  ---- '
    print(row)

# cache results for reuse
save_pickle_atomic({'meta': {'k_grid': KS}, 'results': results}, EVAL_CACHE_PATH)
print(f'\\nsaved {EVAL_CACHE_PATH}')
'''


# Cell 36 — A2.b retrieval dispatch (simplified to bm25 / custom_end_to_end only)
CELL_36_A2_B = '''# A2.b 最终 evidence 选择（检索→分类器统一接口）
#
# 检索路径选择（与下游分类器实验匹配）：
#   'bm25'              → Exp-1 baseline（喂给 DeBERTa-base）
#   'custom_end_to_end' → Exp-2 large / Exp-3 NLI（喂给 large/NLI 分类器，主线）
#
# 输出：final_{train,dev,test}_evidence: Dict[claim_id, List[evidence_id]]
# 这三个字典是下游 A3 训练 / A4 推理的唯一 evidence 来源。

import numpy as np
import faiss
from tqdm.auto import tqdm
from sentence_transformers import CrossEncoder, SentenceTransformer

# ==========================================
# --- 1. MODEL SETUP（仅在 custom_end_to_end 时加载，避免无用显存） ---
# ==========================================

RETRIEVAL_SOURCE = 'custom_end_to_end'   # 'bm25' | 'custom_end_to_end'
FINAL_TOP_K = 5
FORCE_REBUILD_CUSTOM_BGE = False         # flip True 让 build_custom_bge_index 重建缓存

# 三个 split 全走 custom_end_to_end —— train 11k claims × 2-stage 预计 30~60 分钟
# 若需快速验证，可临时改成 ('dev', 'test')
RUN_SPLITS = ('train', 'dev', 'test')

_SPLIT_DATA = {'train': train, 'dev': dev, 'test': test}
final_train_evidence = None
final_dev_evidence   = None
final_test_evidence  = None


def rerank_evidence(claim, candidate_eids, evidence_dict, top_k=5):
    if not candidate_eids:
        return []
    pairs = [[claim, evidence_dict.get(eid, '')] for eid in candidate_eids]
    scores = rerank_model.predict(pairs)
    sorted_indices = np.argsort(scores)[::-1].tolist()
    return [candidate_eids[i] for i in sorted_indices[:top_k]]


def apply_dynamic_reranking(cache_dict, claims_dict, top_k=5, max_candidates=50, desc='Reranking'):
    reranked = {}
    for cid, c in tqdm(claims_dict.items(), desc=desc):
        cand = cache_dict.get(cid, [])[:max_candidates]
        reranked[cid] = rerank_evidence(c['claim_text'], cand, evidence_filtered, top_k=top_k)
    return reranked


def generate_stage1_cache(claims_dict, model, faiss_index, evidence_ids, top_k=50):
    new_cache = {}
    for cid, c in tqdm(claims_dict.items(), desc='Stage 1 Searching'):
        q = model.encode([c['claim_text']], normalize_embeddings=True)
        q = np.array(q).astype('float32')
        _, ix = faiss_index.search(q, top_k)
        new_cache[cid] = [evidence_ids[j] for j in ix[0]]
    return new_cache


def build_custom_bge_index(retriever, evidence_dict, cache_dir, force_rebuild=False):
    """Encode the full corpus with the fine-tuned retriever; cache to disk.

    Cache files under cache_dir:
      custom_bge_evidence_emb.npy  — float32 (N, dim)
      custom_bge_faiss.index       — FAISS IndexFlatIP
      custom_bge_evi_ids.pkl       — evidence id order (must match emb rows)
    """
    emb_path   = f'{cache_dir}/custom_bge_evidence_emb.npy'
    index_path = f'{cache_dir}/custom_bge_faiss.index'
    ids_path   = f'{cache_dir}/custom_bge_evi_ids.pkl'

    evidence_ids   = list(evidence_dict.keys())
    evidence_texts = list(evidence_dict.values())
    expected_dim   = retriever.get_sentence_embedding_dimension()
    n              = len(evidence_ids)

    if not force_rebuild and os.path.exists(emb_path) and os.path.exists(index_path) and os.path.exists(ids_path):
        try:
            cached_ids = load_pickle(ids_path)
            if list(cached_ids) != evidence_ids:
                print('  custom-bge cache: evidence id order mismatch; rebuilding')
            else:
                emb = np.load(emb_path, mmap_mode='r')
                if emb.shape != (n, expected_dim):
                    print(f'  custom-bge cache: shape mismatch {emb.shape} != {(n, expected_dim)}; rebuilding')
                else:
                    idx = faiss.read_index(index_path)
                    if idx.ntotal != n or idx.d != expected_dim:
                        print(f'  custom-bge cache: FAISS dim/ntotal mismatch (ntotal={idx.ntotal}, d={idx.d}); rebuilding')
                    else:
                        print(f'  custom-bge cache HIT -> {index_path} (ntotal={idx.ntotal:,}, dim={idx.d})')
                        return idx, evidence_ids
        except Exception as e:
            print(f'  custom-bge cache load failed ({type(e).__name__}: {e}); rebuilding')

    print(f'Encoding corpus with fine-tuned retriever ({n:,} passages)...')
    emb = retriever.encode(
        evidence_texts, batch_size=128, show_progress_bar=True, normalize_embeddings=True
    )
    emb = np.ascontiguousarray(emb, dtype='float32')
    if not np.isfinite(emb).all():
        raise RuntimeError('custom-bge embeddings contain NaN/Inf; refusing to write cache')

    idx = faiss.IndexFlatIP(emb.shape[1])
    idx.add(emb)

    os.makedirs(cache_dir, exist_ok=True)
    tmp = emb_path + '.tmp.npy'
    np.save(tmp, emb); os.replace(tmp, emb_path)
    tmp = index_path + '.tmp'
    faiss.write_index(idx, tmp); os.replace(tmp, index_path)
    save_pickle_atomic(evidence_ids, ids_path)
    print(f'  saved -> {emb_path} shape={emb.shape}')
    print(f'  saved -> {index_path} ntotal={idx.ntotal:,}')
    return idx, evidence_ids


# ==========================================
# --- 2. FINAL EVIDENCE SELECTION ---
# ==========================================

if RETRIEVAL_SOURCE == 'custom_end_to_end':
    custom_retriever_path = f'{CKPT}/custom-bge-retriever-final'
    custom_reranker_path  = f'{CKPT}/custom-bge-reranker-final'
    assert os.path.isdir(custom_retriever_path), (
        f'缺 {custom_retriever_path}\\n'
        f'请从 Colab Drive 把 custom-bge-retriever-final/ 同步到 ckpt/ (见 README)'
    )
    assert os.path.isdir(custom_reranker_path), (
        f'缺 {custom_reranker_path}\\n'
        f'请从 Colab Drive 把 custom-bge-reranker-final/ 同步到 ckpt/ (见 README)'
    )

    print('Loading Stage 2: FINE-TUNED BGE-Reranker...')
    rerank_model = CrossEncoder(custom_reranker_path, device=DEVICE)
    print('Loading Stage 1: FINE-TUNED BGE-Retriever...')
    retriever_model = SentenceTransformer(custom_retriever_path, device=DEVICE)

    print('\\n--- Initializing FAISS Database (cache-first) ---')
    custom_faiss_index, evidence_ids = build_custom_bge_index(
        retriever_model, evidence_filtered, CACHE,
        force_rebuild=FORCE_REBUILD_CUSTOM_BGE,
    )

    print(f'\\n--- STAGE 1+2 active splits: {RUN_SPLITS} ---')
    for split in ('train', 'dev', 'test'):
        if split not in RUN_SPLITS:
            print(f'  skipping split={split!r} (not in RUN_SPLITS)')
            continue
        print(f'\\n[{split}] STAGE 1: wide-net (Top 50)')
        stage1 = generate_stage1_cache(_SPLIT_DATA[split], retriever_model, custom_faiss_index, evidence_ids, top_k=50)
        print(f'[{split}] STAGE 2: rerank (Top {FINAL_TOP_K})')
        reranked = apply_dynamic_reranking(stage1, _SPLIT_DATA[split], FINAL_TOP_K, max_candidates=50, desc=f'Reranking {split}')
        globals()[f'final_{split}_evidence'] = reranked

elif RETRIEVAL_SOURCE == 'bm25':
    # baseline path: top-5 from BM25 cache（train/dev/test 都需要 bm25_*.pkl）
    for split in ('train', 'dev', 'test'):
        cache_path = f'{CACHE}/bm25_{split}.pkl'
        assert os.path.exists(cache_path), f'缺 {cache_path}，请先跑 cell A1.c'
        globals()[f'final_{split}_evidence'] = load_pickle(cache_path)

else:
    raise ValueError(
        f'Unknown RETRIEVAL_SOURCE: {RETRIEVAL_SOURCE!r} '
        f'(允许值: "bm25" / "custom_end_to_end")'
    )

FINAL_EVIDENCE_SOURCE = RETRIEVAL_SOURCE
TOP_K = FINAL_TOP_K


# --- 3. EVALUATION ---
def f_at_k(candidate_dict, claims, k):
    if candidate_dict is None:
        return None
    fs = []
    for cid, c in claims.items():
        pred = set(candidate_dict.get(cid, [])[:k])
        gold = set(c['evidences'])
        if not pred or not gold:
            fs.append(0.0)
            continue
        tp = len(pred & gold)
        if tp == 0:
            fs.append(0.0)
            continue
        p, r = tp / len(pred), tp / len(gold)
        fs.append(2 * p * r / (p + r))
    return float(np.mean(fs))

print(f'\\nRETRIEVAL_SOURCE = {RETRIEVAL_SOURCE}')
print(f'FINAL_TOP_K      = {FINAL_TOP_K}')
print(f'RUN_SPLITS       = {RUN_SPLITS}')
print(f'\\n--- F@K on dev (final source = {RETRIEVAL_SOURCE}) ---')
if final_dev_evidence is None:
    print('  dev skipped (not in RUN_SPLITS); no F@K to report')
else:
    print(f'{"K":>3}  {"F@K":>8}')
    print('-' * 14)
    for k in [3, 4, 5, 6]:
        f = f_at_k(final_dev_evidence, dev, k)
        marker = ' <- selected' if k == FINAL_TOP_K else ''
        print(f'{k:>3}  {f:>8.3f}{marker}')
'''


CELL_38_A3_1 = '''# A3.1: 标签映射 + 类权重 + 三路 backbone 配置（baseline / large / nli）
from collections import Counter
import torch

LABELS = ['SUPPORTS', 'REFUTES', 'NOT_ENOUGH_INFO', 'DISPUTED']
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}

label_counts = Counter(c['claim_label'] for c in train.values())
print('Train label distribution:')
for l in LABELS:
    n = label_counts.get(l, 0)
    pct = n / len(train) * 100
    print(f'  {l:<18s} {n:>4d}  ({pct:>5.1f}%)')

# inverse-frequency 归一化使 sum = num_classes（保持 loss 量级稳定）
inv = [1.0 / max(label_counts.get(l, 1), 1) for l in LABELS]
s = sum(inv)
CLASS_WEIGHTS = torch.tensor([w / s * len(LABELS) for w in inv], dtype=torch.float)
print(f'\\nClass weights (sum=4): {[f"{w:.2f}" for w in CLASS_WEIGHTS.tolist()]}')


# ----------------------------------------------------------------------
# 三条对照实验路径配置（切换这一行即可重训对照模型）
# ----------------------------------------------------------------------
# 'base'  → Exp-1 Baseline      : BM25 + microsoft/deberta-v3-base (full FT)
# 'large' → Exp-2 Classifier Up : custom BGE+rerank + microsoft/deberta-v3-large + LoRA
# 'nli'   → Exp-3 FINAL         : custom BGE+rerank + MoritzLaurer NLI + LoRA
CLS_BACKBONE_CHOICE = 'nli'

BACKBONE_CONFIG = {
    'base': {
        'name':        'microsoft/deberta-v3-base',
        'use_lora':    False,
        'max_length':  512,
        'batch_size':  8,
        'grad_accum':  2,
        'lr':          2e-5,
        'epochs':      3,
        'retrieval':   'bm25',
        'pred_prefix': 'baseline-',
        'ckpt_name':   'cls_best_base.pt',
    },
    'large': {
        'name':        'microsoft/deberta-v3-large',
        'use_lora':    True,
        'max_length':  512,
        'batch_size':  4,
        'grad_accum':  4,
        'lr':          2e-4,
        'epochs':      3,
        'retrieval':   'custom_end_to_end',
        'pred_prefix': 'large-',
        'ckpt_name':   'cls_best_large.pt',
    },
    'nli': {
        'name':        'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli',
        'use_lora':    True,
        'max_length':  512,
        'batch_size':  4,
        'grad_accum':  4,
        'lr':          2e-4,
        'epochs':      3,
        'retrieval':   'custom_end_to_end',
        'pred_prefix': 'nli-',
        'ckpt_name':   'cls_best_nli.pt',
    },
}
assert CLS_BACKBONE_CHOICE in BACKBONE_CONFIG, f'未知 CLS_BACKBONE_CHOICE={CLS_BACKBONE_CHOICE!r}'
CFG = BACKBONE_CONFIG[CLS_BACKBONE_CHOICE]
print(f'\\n[A3.1] CLS_BACKBONE_CHOICE = {CLS_BACKBONE_CHOICE!r}')
print(f'        backbone  = {CFG["name"]}')
print(f'        retrieval = {CFG["retrieval"]}')
print(f'        LoRA      = {CFG["use_lora"]}  max_len={CFG["max_length"]}  bs={CFG["batch_size"]}  ga={CFG["grad_accum"]}')
'''


CELL_39_A3_2 = '''# A3.2: OOP 类 — ClaimDS / BaselineClassifier / Trainer
# 三个 backbone 共用同一套类，由 cell A3.1 的 CFG 字典驱动差异：
#   - base : 全量微调，不开 LoRA
#   - large/nli : LoRA r=16, α=32（target=query_proj/value_proj）
# 所有 backbone 的 cls head 都是新训的 nn.Linear(h, 4)，
# 即使 MoritzLaurer 原本是 3-class NLI 也不会复用其 head — 4 类 head 完全是新初始化的。

import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

CLS_BACKBONE = CFG['name']

class ClaimDS(Dataset):
    """拼 [CLS] claim [SEP] evi_1 [SEP] evi_2 ... [SEP] 喂分类器。
    use_gold=True  : 训练前期，evidence 来自 c['evidences']（gold supervision）
    use_gold=False : 训练后期 / 推理，evidence 来自 evi_dict[cid][:top_k]
    """
    def __init__(self, claims, evidence_filtered, evi_dict=None, use_gold=True, top_k=5,
                 tokenizer_name=None, max_length=None):
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name or CLS_BACKBONE, use_fast=False)
        self.max_length = max_length or CFG['max_length']
        self.items = []
        sep = ' [SEP] '
        for cid, c in claims.items():
            if use_gold:
                evis = c.get('evidences', [])
            else:
                evis = (evi_dict.get(cid, []) if evi_dict else [])[:top_k]
            evi_text = sep.join(evidence_filtered.get(e, '') for e in evis) or 'no evidence'
            label = LABEL2ID.get(c.get('claim_label', 'NOT_ENOUGH_INFO'), LABEL2ID['NOT_ENOUGH_INFO'])
            self.items.append((c['claim_text'], evi_text, label, cid))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        claim, evi, label, _ = self.items[i]
        enc = self.tok(claim, evi, truncation=True, max_length=self.max_length,
                       padding='max_length', return_tensors='pt')
        return {
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label':          torch.tensor(label, dtype=torch.long),
        }


class BaselineClassifier(nn.Module):
    """DeBERTa backbone + 单层 cls head；large/nli 走 LoRA, base 走全量微调。"""
    def __init__(self, backbone=None, n_classes=4, dropout=0.15,
                 use_lora=None, lora_r=16, lora_alpha=32):
        super().__init__()
        backbone = backbone or CLS_BACKBONE
        use_lora = CFG['use_lora'] if use_lora is None else use_lora
        self.encoder = AutoModel.from_pretrained(backbone)

        if use_lora:
            print(f'Applying LoRA PEFT to {backbone}... (r={lora_r}, alpha={lora_alpha})')
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=['query_proj', 'value_proj'],
                lora_dropout=0.1,
                bias='none',
            )
            self.encoder = get_peft_model(self.encoder, peft_config)
            self.encoder.print_trainable_parameters()
        else:
            print(f'Full fine-tuning {backbone} (LoRA disabled)')

        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.cls = nn.Linear(h, n_classes)   # 新训的 4-class head（不复用 backbone 原 NLI head）

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1)   # mean pooling
        return self.cls(self.dropout(pooled))


class Trainer:
    """fp16 (cuda) + class-weighted CE + label smoothing + cosine LR + best-ckpt on dev_acc."""
    def __init__(self, model, train_ds, dev_ds, class_weights,
                 device='cuda', lr=None, epochs=None, batch_size=None, grad_accum=None,
                 ckpt_path=None, label_smoothing=0.1, max_grad_norm=1.0):
        from tqdm.auto import tqdm
        self.tqdm = tqdm
        self.device = device
        self.model = model.to(device)
        batch_size = batch_size or CFG['batch_size']
        grad_accum = grad_accum or CFG['grad_accum']
        lr = lr or CFG['lr']
        epochs = epochs or CFG['epochs']

        self.train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.dev_loader   = DataLoader(dev_ds,   batch_size=batch_size, shuffle=False)

        # 优化器只跟踪 requires_grad=True 的参数（LoRA + cls head；base 则全部参数）
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        steps = max(1, len(self.train_loader) // grad_accum * epochs)
        warmup = max(1, steps // 10)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, warmup, steps)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device), label_smoothing=label_smoothing,
        )
        self.use_amp = (device == 'cuda')
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.epochs = epochs
        self.grad_accum = grad_accum
        self.max_grad_norm = max_grad_norm
        self.ckpt_path = ckpt_path

    def _forward_loss(self, batch):
        ids  = batch['input_ids'].to(self.device)
        mask = batch['attention_mask'].to(self.device)
        y    = batch['label'].to(self.device)
        if self.use_amp:
            with torch.cuda.amp.autocast():
                logits = self.model(ids, mask)
                loss = self.criterion(logits, y)
        else:
            logits = self.model(ids, mask)
            loss = self.criterion(logits, y)
        return loss

    def train(self):
        best = 0.0
        for ep in range(self.epochs):
            self.model.train()
            total = 0.0
            self.optimizer.zero_grad()
            for step, batch in enumerate(self.tqdm(self.train_loader, desc=f'ep{ep}')):
                loss = self._forward_loss(batch) / self.grad_accum
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (step + 1) % self.grad_accum == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()
                total += loss.item() * self.grad_accum
            avg = total / max(1, len(self.train_loader))
            acc = self.evaluate()
            print(f'  epoch {ep}: loss={avg:.3f}  dev_acc={acc:.3f}')
            if acc > best and self.ckpt_path:
                best = acc
                torch.save(self.model.state_dict(), self.ckpt_path)
                print(f'    ✓ saved → {self.ckpt_path}')
        return best

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        ok = total = 0
        for batch in self.dev_loader:
            ids  = batch['input_ids'].to(self.device)
            mask = batch['attention_mask'].to(self.device)
            y    = batch['label'].to(self.device)
            pred = self.model(ids, mask).argmax(-1)
            ok    += (pred == y).sum().item()
            total += y.size(0)
        return ok / max(1, total)
'''


CELL_40_A3_TRAIN_ENTRY = '''# A3 训练入口 — 根据 CFG 选择 backbone / ckpt / 训练数据源
# 训练前确认 RETRIEVAL_SOURCE 与 CFG['retrieval'] 一致，防止 baseline 误用 custom 检索之类

CLS_BEST_PATH  = f'{CKPT}/{CFG["ckpt_name"]}'
FINAL_CKPT_PATH = CLS_BEST_PATH
CLS_FINAL_PATH = FINAL_CKPT_PATH

# 进入分类器训练前释放 retrieval-only 大对象（每条实验路径都跑一次）
release_large_objects(
    'bm25', 'evidence', 'bge_index_obj', 'bge_embeddings', 'bge_corpus_emb',
)

# 一致性检查：训练用的检索源必须和 BACKBONE_CONFIG 声明的一致
assert RETRIEVAL_SOURCE == CFG['retrieval'], (
    f"RETRIEVAL_SOURCE={RETRIEVAL_SOURCE!r} 与 BACKBONE_CONFIG[{CLS_BACKBONE_CHOICE!r}]"
    f"['retrieval']={CFG['retrieval']!r} 不一致 — 请回 cell A2.b 改 RETRIEVAL_SOURCE 后重跑"
)
assert final_train_evidence is not None, (
    'final_train_evidence is None — 请回 cell A2.b 把 train 加入 RUN_SPLITS 后重跑'
)
assert final_dev_evidence is not None, 'final_dev_evidence is None — 同上'

print(f'[A3 training] choice={CLS_BACKBONE_CHOICE!r}  ckpt={CLS_BEST_PATH}')
print(f'              backbone={CFG["name"]}')
print(f'              train evidence source = {RETRIEVAL_SOURCE}')

# 两阶段训练：
#   Stage 1 (gold supervision)   — evi 来自 c['evidences']，纯学分类 head + 编码器
#   Stage 2 (retrieved supervision) — evi 来自 final_*_evidence，让模型适应检索噪声
RUN_STAGE_2 = True   # 设 False 可只跑 Stage 1（更快，但 dev_acc 略低）

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE = {DEVICE}')

# Stage 1：gold-supervised
print('\\n=== Stage 1: gold-supervised training ===')
train_ds = ClaimDS(train, evidence_filtered, use_gold=True)
dev_ds   = ClaimDS(dev,   evidence_filtered, evi_dict=final_dev_evidence, use_gold=False, top_k=FINAL_TOP_K)
model    = BaselineClassifier()
trainer  = Trainer(model, train_ds, dev_ds, CLASS_WEIGHTS, device=DEVICE, ckpt_path=CLS_BEST_PATH)
best_s1  = trainer.train()
print(f'Stage 1 best dev_acc = {best_s1:.4f}')

if RUN_STAGE_2:
    print('\\n=== Stage 2: retrieved-supervised fine-tuning ===')
    model.load_state_dict(torch.load(CLS_BEST_PATH, map_location=DEVICE))
    train_ds2 = ClaimDS(train, evidence_filtered, evi_dict=final_train_evidence, use_gold=False, top_k=FINAL_TOP_K)
    trainer2  = Trainer(model, train_ds2, dev_ds, CLASS_WEIGHTS, device=DEVICE,
                        ckpt_path=CLS_BEST_PATH, epochs=max(1, CFG['epochs'] // 2))
    best_s2 = trainer2.train()
    print(f'Stage 2 best dev_acc = {best_s2:.4f}')
'''


CELL_42_A4_HEADER = '''## A4: 端到端预测 + 三路对照

把 A3 训好的分类器接上 retrieval Top-K，得到 dev/test 预测。

**三条对照路径**（在 cell A3.1 把 `CLS_BACKBONE_CHOICE` 切到对应值后 Run All）：

| 路径 | 检索 | 分类器 | 输出文件 |
|---|---|---|---|
| Exp-1 baseline | BM25 top-5 | DeBERTa-v3-base | `baseline-{dev,test}-predictions.json` |
| Exp-2 large | custom BGE+rerank top-5 | DeBERTa-v3-large + LoRA | `large-{dev,test}-predictions.json` |
| **Exp-3 NLI（最终）** | custom BGE+rerank top-5 | MoritzLaurer NLI + LoRA | `nli-{dev,test}-predictions.json` <br>+ A4.3 per-evi: `nli-per-evi-{dev,test}-predictions.json` <br>+ A4.5 hybrid routing: `hybrid-{dev,test}-predictions.json` |

A4.1 是拼接式推理（一次 forward 看 5 条 evidence），所有三条路径都跑；
A4.3 / A4.5 只在 NLI 路径下启用（per-evidence NLI + routing 与 NLI 预训练绑定）。
A6 cell 会读取三套预测自动出对照表。'''


CELL_43_A4_1 = '''# A4.1: 端到端 predict + 保存 dev/test predictions
# 输出文件名由 CFG["pred_prefix"] 决定：baseline- / large- / nli-
import json, torch
from transformers import AutoTokenizer
from tqdm.auto import tqdm

# 加载当前 CFG 对应的 checkpoint
model.load_state_dict(torch.load(FINAL_CKPT_PATH, map_location=DEVICE))
model.eval()


def predict(claims, evidence_candidates, evidence_filtered, k=None,
            tokenizer_name=None, max_length=None, device=DEVICE):
    if k is None:
        k = FINAL_TOP_K
    tokenizer_name = tokenizer_name or CLS_BACKBONE
    max_length = max_length or CFG['max_length']
    tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
    sep = ' [SEP] '
    out = {}
    for cid, c in tqdm(claims.items(), desc='predict'):
        evis = list(evidence_candidates.get(cid, []))[:k]
        if not evis:
            evis = list(evidence_candidates.get(cid, []))[:1]
        if not evis:
            evis = [next(iter(evidence_filtered))]   # eval.py 要求至少 1 条
        evi_text = sep.join(evidence_filtered.get(eid, '') for eid in evis) or 'no evidence'
        enc = tok(c['claim_text'], evi_text, truncation=True, max_length=max_length,
                  padding='max_length', return_tensors='pt')
        with torch.no_grad():
            ids  = enc['input_ids'].to(device)
            mask = enc['attention_mask'].to(device)
            if device == 'cuda':
                with torch.cuda.amp.autocast():
                    pred = model(ids, mask).argmax(-1).item()
            else:
                pred = model(ids, mask).argmax(-1).item()
        out[cid] = {
            'claim_text': c['claim_text'],
            'claim_label': ID2LABEL[pred],
            'evidences': evis[:6],
        }
    return out


PREFIX = CFG['pred_prefix']   # 'baseline-' | 'large-' | 'nli-'
dev_pred_path  = f'{ROOT}/{PREFIX}dev-predictions.json'
test_pred_path = f'{ROOT}/{PREFIX}test-claims-predictions.json'

print(f'Predicting dev with retrieval={RETRIEVAL_SOURCE} Top-{FINAL_TOP_K} from ckpt {FINAL_CKPT_PATH}')
dev_pred = predict(dev, final_dev_evidence, evidence_filtered, k=FINAL_TOP_K)
with open(dev_pred_path, 'w') as f:
    json.dump(dev_pred, f, indent=2)
print(f'\\nSaved {dev_pred_path}  ({len(dev_pred)} claims)')

test_pred = predict(test, final_test_evidence, evidence_filtered, k=FINAL_TOP_K)
with open(test_pred_path, 'w') as f:
    json.dump(test_pred, f, indent=2)
print(f'Saved {test_pred_path}  ({len(test_pred)} claims)')

sample_cid = next(iter(dev_pred))
print(f'\\nSample [{sample_cid}]:')
print(f"  claim    : {dev_pred[sample_cid]['claim_text'][:80]}...")
print(f"  pred     : {dev_pred[sample_cid]['claim_label']}")
print(f"  evidence : {dev_pred[sample_cid]['evidences'][:3]}...")
'''


CELL_46_A4_3 = '''# A4.3: per-evidence NLI 推理 + 聚合（仅 NLI 路径有效，base/large 跳过）
if CLS_BACKBONE_CHOICE != 'nli':
    print(f'[A4.3] skip per-evidence NLI inference (CHOICE={CLS_BACKBONE_CHOICE!r}, '
          f'only meaningful for nli backbone)')
else:
    import json, numpy as np, torch
    from collections import Counter
    from transformers import AutoTokenizer
    from tqdm.auto import tqdm

    # 确保用最终 NLI checkpoint
    model.load_state_dict(torch.load(FINAL_CKPT_PATH, map_location=DEVICE))
    model.eval()

    _NLI_TOK = AutoTokenizer.from_pretrained(CLS_BACKBONE, use_fast=False)
    SUP = LABEL2ID['SUPPORTS']
    REF = LABEL2ID['REFUTES']
    NEI = LABEL2ID['NOT_ENOUGH_INFO']
    DIS = LABEL2ID['DISPUTED']

    NLI_MAX_LEN = 320  # 单条 evidence，比拼接版短，省显存

    @torch.no_grad()
    def _probs_for_pair(claim_text, evi_text, max_length=NLI_MAX_LEN):
        enc = _NLI_TOK(claim_text, evi_text, truncation=True, max_length=max_length,
                       padding='max_length', return_tensors='pt')
        ids  = enc['input_ids'].to(DEVICE)
        mask = enc['attention_mask'].to(DEVICE)
        if DEVICE == 'cuda':
            with torch.cuda.amp.autocast():
                logits = model(ids, mask)
        else:
            logits = model(ids, mask)
        return torch.softmax(logits.float(), dim=-1).squeeze(0).cpu().numpy()

    def _aggregate(per_evi_probs, tau_disp):
        """list[np.ndarray(4)] -> (label_id, mean_probs[4])"""
        if not per_evi_probs:
            v = np.zeros(4); v[NEI] = 1.0
            return NEI, v
        arr = np.stack(per_evi_probs, axis=0)
        mean_p = arr.mean(axis=0)
        max_sup = float(arr[:, SUP].max())
        max_ref = float(arr[:, REF].max())
        if max_sup >= tau_disp and max_ref >= tau_disp:
            return DIS, mean_p
        return int(mean_p.argmax()), mean_p

    def _per_evi_cache(claims, evi_dict, k):
        cache = {}
        for cid, c in tqdm(claims.items(), desc='nli-cache'):
            evis = list(evi_dict.get(cid, []))[:k]
            if not evis:
                evis = [next(iter(evidence_filtered))]
            probs = []
            for eid in evis:
                evi_text = evidence_filtered.get(eid, '') or 'no evidence'
                probs.append(_probs_for_pair(c['claim_text'], evi_text))
            cache[cid] = (evis, probs)
        return cache

    print('Caching per-evidence probs on dev (one forward per evidence)...')
    dev_cache = _per_evi_cache(dev, final_dev_evidence, k=FINAL_TOP_K)

    print('\\n=== Sweep DISPUTED conflict threshold τ on dev ===')
    def _eval_tau(tau):
        correct = 0
        cm = Counter()
        for cid, (_, probs) in dev_cache.items():
            gold = dev[cid]['claim_label']
            pid, _m = _aggregate(probs, tau_disp=tau)
            plabel = ID2LABEL[pid]
            cm[(gold, plabel)] += 1
            if plabel == gold:
                correct += 1
        return correct / len(dev_cache), cm

    best_tau, best_acc, best_cm = 1.01, -1.0, None
    TAU_GRID = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 1.01]
    for tau in TAU_GRID:
        acc, cm = _eval_tau(tau)
        marker = ''
        if acc > best_acc:
            best_acc, best_tau, best_cm = acc, tau, cm
            marker = ' *'
        print(f'  τ={tau:>4.2f}  dev_acc={acc:.4f}{marker}')
    print(f'\\nBest τ = {best_tau}  dev_acc = {best_acc:.4f}')

    # 输出文件名加 nli-per-evi- 前缀，避免和 A4.1 拼接结果重名
    nli_dev_pred_path  = f'{ROOT}/nli-per-evi-dev-predictions.json'
    nli_test_pred_path = f'{ROOT}/nli-per-evi-test-claims-predictions.json'

    dev_nli_pred = {}
    for cid, (evis, probs) in dev_cache.items():
        pid, _m = _aggregate(probs, tau_disp=best_tau)
        dev_nli_pred[cid] = {
            'claim_text': dev[cid]['claim_text'],
            'claim_label': ID2LABEL[pid],
            'evidences': evis[:6],
        }
    with open(nli_dev_pred_path, 'w') as f:
        json.dump(dev_nli_pred, f, indent=2)
    print(f'Saved {nli_dev_pred_path}  ({len(dev_nli_pred)} claims)')

    print('\\nPredicting test with NLI aggregation...')
    test_cache = _per_evi_cache(test, final_test_evidence, k=FINAL_TOP_K)
    test_nli_pred = {}
    for cid, (evis, probs) in test_cache.items():
        pid, _m = _aggregate(probs, tau_disp=best_tau)
        test_nli_pred[cid] = {
            'claim_text': test[cid]['claim_text'],
            'claim_label': ID2LABEL[pid],
            'evidences': evis[:6],
        }
    with open(nli_test_pred_path, 'w') as f:
        json.dump(test_nli_pred, f, indent=2)
    print(f'Saved {nli_test_pred_path}  ({len(test_nli_pred)} claims)')

    print('\\n=== Dev confusion matrix (rows=gold, cols=pred) @ best τ ===')
    header = ' ' * 18 + '  '.join(f'{l[:5]:>5s}' for l in LABELS)
    print(header)
    for gl in LABELS:
        row = [best_cm.get((gl, pl), 0) for pl in LABELS]
        print(f'  {gl:<16s}' + '  '.join(f'{v:>5d}' for v in row))

    NLI_BEST_TAU = best_tau
'''


CELL_49_A4_5 = '''# A4.5: Hybrid routing — baseline-NEI ∪ NLI-(SUP/REF)（仅 NLI 路径有效）
if CLS_BACKBONE_CHOICE != 'nli':
    print(f'[A4.5] skip hybrid routing (CHOICE={CLS_BACKBONE_CHOICE!r}, requires nli + A4.3 outputs)')
else:
    import json, subprocess
    from collections import Counter

    def _score(pred_path, gold_dict):
        pred = json.load(open(pred_path))
        cm = Counter()
        correct = total = 0
        for cid, c in gold_dict.items():
            if cid not in pred:
                continue
            gold = c['claim_label']
            p = pred[cid]['claim_label']
            cm[(gold, p)] += 1
            if p == gold:
                correct += 1
            total += 1
        return correct / max(1, total), cm, total

    def _print_cm(cm, title):
        print(f'\\n{title}')
        header = ' ' * 18 + '  '.join(f'{l[:5]:>5s}' for l in LABELS)
        print(header)
        for gl in LABELS:
            row = [cm.get((gl, pl), 0) for pl in LABELS]
            print(f'  {gl:<16s}' + '  '.join(f'{v:>5d}' for v in row))

    def _route(base, nli, claims):
        out = {}
        for cid, c in claims.items():
            bp = base.get(cid)
            np_ = nli.get(cid)
            if bp and bp['claim_label'] == 'NOT_ENOUGH_INFO':
                chosen, evis = bp['claim_label'], bp['evidences']
            elif np_ is not None:
                chosen, evis = np_['claim_label'], np_['evidences']
            elif bp is not None:
                chosen, evis = bp['claim_label'], bp['evidences']
            else:
                chosen, evis = 'NOT_ENOUGH_INFO', []
            out[cid] = {'claim_text': c['claim_text'], 'claim_label': chosen, 'evidences': evis}
        return out

    # 读 A4.1 (nli-*-predictions.json) 和 A4.3 (nli-per-evi-*-predictions.json)
    base_dev = json.load(open(f'{ROOT}/nli-dev-predictions.json'))
    nli_dev  = json.load(open(f'{ROOT}/nli-per-evi-dev-predictions.json'))
    hybrid_dev = _route(base_dev, nli_dev, dev)

    hybrid_dev_path  = f'{ROOT}/hybrid-dev-predictions.json'
    hybrid_test_path = f'{ROOT}/hybrid-test-claims-predictions.json'

    with open(hybrid_dev_path, 'w') as f:
        json.dump(hybrid_dev, f, indent=2)
    print(f'Saved {hybrid_dev_path}  ({len(hybrid_dev)} claims)')

    base_test = json.load(open(f'{ROOT}/nli-test-claims-predictions.json'))
    nli_test  = json.load(open(f'{ROOT}/nli-per-evi-test-claims-predictions.json'))
    hybrid_test = _route(base_test, nli_test, test)
    with open(hybrid_test_path, 'w') as f:
        json.dump(hybrid_test, f, indent=2)
    print(f'Saved {hybrid_test_path}  ({len(hybrid_test)} claims)')

    # 三条 NLI 路径对照（dev）
    acc_b, cm_b, _ = _score(f'{ROOT}/nli-dev-predictions.json',         dev)
    acc_n, cm_n, _ = _score(f'{ROOT}/nli-per-evi-dev-predictions.json', dev)
    acc_h, cm_h, _ = _score(hybrid_dev_path,                            dev)
    print(f'\\nNLI cat (A4.1)     A = {acc_b:.4f}')
    print(f'NLI per-evi (A4.3) A = {acc_n:.4f}   △ vs cat = {acc_n-acc_b:+.4f}')
    print(f'Hybrid (A4.5)      A = {acc_h:.4f}   △ vs cat = {acc_h-acc_b:+.4f}')

    _print_cm(cm_h, '=== Hybrid confusion matrix ===')

    print('\\n--- eval.py: hybrid-dev ---')
    r = subprocess.run(['python', f'{DATA}/eval.py',
                        '--predictions', hybrid_dev_path,
                        '--groundtruth', f'{DATA}/dev-claims.json'],
                       capture_output=True, text=True)
    print(r.stdout if r.returncode == 0 else r.stderr)
'''


# ----------------------------------------------------------------------
# New cells (A6 comparison + A5 smoke test)
# ----------------------------------------------------------------------

CELL_NEW_A6_MARKDOWN = '''## A6: 三路对照实验汇总表

按 CLS_BACKBONE_CHOICE = `'base'` / `'large'` / `'nli'` 各跑一次后，本 cell 自动读三套 dev predictions JSON、调 eval.py，画出对照表。某一行 `–` 表示对应实验还没产出。'''


CELL_NEW_A6_CODE = '''# A6: 读 baseline-/large-/nli- 三套 dev predictions，调 eval.py 出对照表
import os, json, subprocess
from collections import OrderedDict

EXPERIMENTS = OrderedDict([
    ('Exp-1 Baseline (BM25 + DeBERTa-base)',          'baseline-dev-predictions.json'),
    ('Exp-2 + Retrieval upgrade + DeBERTa-large',     'large-dev-predictions.json'),
    ('Exp-3 Final: NLI backbone (A4.1 cat)',          'nli-dev-predictions.json'),
    ('Exp-3 Final: NLI per-evidence (A4.3)',          'nli-per-evi-dev-predictions.json'),
    ('Exp-3 Final: Hybrid routing (A4.5) [采用]',     'hybrid-dev-predictions.json'),
])

def _parse_eval(stdout):
    parsed = {}
    for line in stdout.splitlines():
        for key, tag in [('Evidence Retrieval F-score', 'F'),
                         ('Claim Classification Accuracy', 'A'),
                         ('Harmonic Mean', 'Hmean')]:
            if key in line and '=' in line:
                try:
                    parsed[tag] = float(line.rsplit('=', 1)[-1].strip())
                except ValueError:
                    pass
    return parsed if {'F', 'A', 'Hmean'} <= parsed.keys() else None


def _eval(pred_path):
    if not os.path.exists(pred_path):
        return None
    r = subprocess.run(['python', f'{DATA}/eval.py',
                        '--predictions', pred_path,
                        '--groundtruth', f'{DATA}/dev-claims.json'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return _parse_eval(r.stdout)


print('| Experiment | F | A | Hmean |')
print('|---|---|---|---|')
for name, fname in EXPERIMENTS.items():
    res = _eval(f'{ROOT}/{fname}')
    if res is None:
        print(f'| {name} | – | – | – |')
    else:
        print(f'| {name} | {res["F"]:.3f} | {res["A"]:.3f} | {res["Hmean"]:.4f} |')
print('\\n（缺失行：把 cell A3.1 的 CLS_BACKBONE_CHOICE 切到对应值后 Run All 即可补齐）')
'''


CELL_NEW_A5_MARKDOWN = '''## A5: 端到端 smoke test

8 条 dev claim 走完整 retrieve → rerank → classify → eval 流程，2 分钟内验证 pipeline 没坏。
跑这一 cell 之前需要：cell A2.b 已完成（保证 retriever_model / rerank_model / custom_faiss_index 在内存里），且 cell A4.1 的 `predict` 函数已定义。'''


CELL_NEW_A5_CODE = '''# A5: end-to-end smoke test —— 8 条 dev claim 验证整条 pipeline
import os, json, subprocess

REQUIRED_PATHS = {
    'data':       [f'{DATA}/train-claims.json', f'{DATA}/dev-claims.json',
                   f'{DATA}/test-claims-unlabelled.json', f'{DATA}/evidence.json',
                   f'{DATA}/eval.py'],
    'cache':      [f'{CACHE}/evidence_filtered.pkl', f'{CACHE}/bm25.pkl'],
    'checkpoint': [FINAL_CKPT_PATH,
                   f'{CKPT}/custom-bge-retriever-final',
                   f'{CKPT}/custom-bge-reranker-final'],
}

def _check(group, paths):
    missing = [p for p in paths if not os.path.exists(p)]
    print(f'  {group:11s}: {"OK" if not missing else f"MISSING: {missing}"}')
    return not missing

print('=== A5.1 文件就位检查 ===')
all_ok = all(_check(g, ps) for g, ps in REQUIRED_PATHS.items())
if not all_ok:
    print('\\n⚠️ 关键文件缺失 —— 请按 README "Drive checkpoint 同步" 章节补齐后重跑')
else:
    print('\\n=== A5.2 8 条 dev claim 走完整 pipeline ===')
    dev_mini = dict(list(dev.items())[:8])

    # 检索：动态跑 stage-1 (50) + stage-2 (5)，不依赖任何 hybrid_*.pkl
    mini_stage1 = generate_stage1_cache(dev_mini, retriever_model, custom_faiss_index, evidence_ids, top_k=50)
    mini_evi    = apply_dynamic_reranking(mini_stage1, dev_mini, top_k=FINAL_TOP_K, max_candidates=50, desc='A5 mini-rerank')

    # 分类 (A4.1 拼接)
    mini_pred = predict(dev_mini, mini_evi, evidence_filtered, k=FINAL_TOP_K)
    mini_path = f'{ROOT}/a5_smoke_dev.json'
    with open(mini_path, 'w') as f:
        json.dump(mini_pred, f, indent=2)
    print(f'  saved {mini_path}  ({len(mini_pred)} claims)')

    r = subprocess.run(['python', f'{DATA}/eval.py',
                        '--predictions', mini_path,
                        '--groundtruth', f'{DATA}/dev-claims.json'],
                       capture_output=True, text=True)
    if r.returncode == 0:
        print(r.stdout)
        print('=== A5 PASS ===')
    else:
        print(f'  eval.py FAILED: {r.stderr}')
        print('=== A5 FAIL ===')
'''


# ----------------------------------------------------------------------
# Apply
# ----------------------------------------------------------------------

def main():
    with open(NB_PATH) as f:
        nb = json.load(f)

    # --- replacements ---
    replacements = [
        ('14267b8a',  CELL_7_CACHE_META,    'code'),
        ('23320e3f',  CELL_23_A1_HEADER,    'markdown'),
        ('ee509c39',  CELL_24_A1_A_EVAL,    'code'),
        ('e50ff75d',  CELL_36_A2_B,         'code'),
        ('14b2d465',  CELL_38_A3_1,         'code'),
        ('ba911328',  CELL_39_A3_2,         'code'),
        ('bIGHVIH4wcXn', CELL_40_A3_TRAIN_ENTRY, 'code'),
        ('c95d8f69',  CELL_42_A4_HEADER,    'markdown'),
        ('ac06df41',  CELL_43_A4_1,         'code'),
        ('bea07c65',  CELL_46_A4_3,         'code'),
        ('7f9110d6',  CELL_49_A4_5,         'code'),
    ]
    for cid, src, ctype in replacements:
        i, cell = find_cell(nb, cid)
        assert cell['cell_type'] == ctype, (
            f'cell {cid} expected type {ctype}, got {cell["cell_type"]}')
        replace_source(cell, src)
        print(f'  replaced cell {i:>3d} (id={cid})  → {len(src.splitlines())} lines')

    # --- deletions: cell 22 (75c2eca0), cell 25 (e3abce25), cell 47 (5346cab5 - A4.4 old) ---
    # A4.4 (5346cab5) is the old "NLI agg vs baseline" eval; A4.5 (now guarded) already calls eval.py
    # and prints accuracy comparison, so A4.4 is redundant. Drop it.
    for cid in ['75c2eca0', 'e3abce25', '5346cab5']:
        i, _ = find_cell(nb, cid)
        print(f'  deleting cell {i:>3d} (id={cid})')
        del nb['cells'][i]

    # --- inserts: A6 markdown + code, A5 markdown + code, BEFORE the OOP cell (id=b59f5ce9) ---
    i_oop, _ = find_cell(nb, 'b59f5ce9')

    def code_cell(src):
        return {
            'cell_type': 'code',
            'id': make_id(),
            'metadata': {},
            'execution_count': None,
            'outputs': [],
            'source': src.splitlines(keepends=True),
        }

    def md_cell(src):
        return {
            'cell_type': 'markdown',
            'id': make_id(),
            'metadata': {},
            'source': src.splitlines(keepends=True),
        }

    new_cells = [
        md_cell(CELL_NEW_A5_MARKDOWN),
        code_cell(CELL_NEW_A5_CODE),
        md_cell(CELL_NEW_A6_MARKDOWN),
        code_cell(CELL_NEW_A6_CODE),
    ]
    # insert before OOP cell, in order
    for offset, cell in enumerate(new_cells):
        nb['cells'].insert(i_oop + offset, cell)
        print(f'  inserted new {cell["cell_type"]} cell at index {i_oop + offset}')

    # --- write back ---
    with open(NB_PATH, 'w') as f:
        json.dump(nb, f, indent=2)
    print(f'\\nDone. Total cells: {len(nb["cells"])}')


if __name__ == '__main__':
    main()
