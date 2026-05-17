"""Translate all Chinese commentary in the project notebook to English.

Strategy:
  - Cells I rewrote during integration (5, 7, 20, 22-24, 34-38, 40-50) → full English rewrite
  - Cells originally authored by the team (4, 6, 8-12, 18, 21) → translate only
    comments and docstrings; keep all code identical
"""
import json
import re

NB = 'Group_051__COMP90042_Project_2026.ipynb'


def replace_cell(cells, cell_id, new_src):
    for c in cells:
        if c.get('id') == cell_id:
            c['source'] = new_src.splitlines(keepends=True)
            if c['source'] and c['source'][-1].endswith('\n'):
                c['source'][-1] = c['source'][-1].rstrip('\n')
            return
    raise SystemExit(f'cell {cell_id} not found')


# ======================================================================
# Cells I rewrote during integration — fresh English drafts
# ======================================================================

CELL_5 = '''# 1.1.b Data-availability sanity check (team-shared caches: avoid re-running
# heavy steps for every contributor).
#
# This cell only checks whether the expected files are present; loading and
# validation happen later via the cache helpers.
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
    # Custom BGE retriever cache (built by A2.b on first run, ~3.5 GB)
    'cache/custom_bge_evidence_emb.npy': 'fine-tuned BGE evidence embeddings',
    'cache/custom_bge_faiss.index'    : 'fine-tuned BGE FAISS index',
    'cache/custom_bge_evi_ids.pkl'    : 'fine-tuned BGE evidence id order',
}

print(f'Checking ROOT = {ROOT}')
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
        f'Missing required files: {missing_required}\\n'
        f'Make sure the Drive shortcut is mounted, or place data/ next to '
        f'{ROOT} when running locally.'
    )
print('-' * 70)
print('Required data is in place')

print('\\nQuick existence check of official caches (presence only — '
      'validation happens later)')
print('-' * 70)
missing_cache = []
for rel, desc in OFFICIAL_CACHE.items():
    p = os.path.join(ROOT, rel)
    if os.path.exists(p):
        print(f'  ok {rel:<34s} {os.path.getsize(p)/1e6:>8.1f} MB  ({desc})')
    else:
        print(f'  missing {rel:<34s} ({desc})')
        missing_cache.append(rel)

# Fast-path summary: cache readiness per experiment route
BM25_CACHE_RELS = ['cache/bm25_dev.pkl', 'cache/bm25_train.pkl', 'cache/bm25_test.pkl']
CUSTOM_BGE_RELS = ['cache/custom_bge_evidence_emb.npy', 'cache/custom_bge_faiss.index', 'cache/custom_bge_evi_ids.pkl']
ALL_BM25_FILES_PRESENT       = all(os.path.exists(os.path.join(ROOT, rel)) for rel in BM25_CACHE_RELS)
ALL_CUSTOM_BGE_FILES_PRESENT = all(os.path.exists(os.path.join(ROOT, rel)) for rel in CUSTOM_BGE_RELS)
print('\\nFast-path summary:')
print(f'  Exp-1 baseline (BM25) cache present       : {ALL_BM25_FILES_PRESENT}')
print(f'  Exp-2/3 custom BGE corpus cache present   : {ALL_CUSTOM_BGE_FILES_PRESENT}')
print('                   (if False, cell A2.b rebuilds on first run, ~30 minutes)')
'''


CELL_7 = '''# 1.2.b Cache helpers (must be defined before any heavy cache is consumed)
import os, pickle, gc, time

# Experiment routes (three lines):
#   Exp-1 baseline : BM25              -> DeBERTa-v3-base
#   Exp-2 large    : custom BGE+rerank -> DeBERTa-v3-large + LoRA
#   Exp-3 NLI      : custom BGE+rerank -> MoritzLaurer NLI + LoRA    <-- final choice
# We removed the Hybrid BM25-OR-BGE route to avoid duplicating custom_end_to_end.

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
        # raw BGE — used for the A1.a retrieval ablation (vs. the fine-tuned version)
        'version': 1,
        'model': 'BAAI/bge-base-en-v1.5',
        'max_seq_len': 384,
        'embedding_dim': 768,
    },
    'custom_bge': {
        # Fine-tuned BGE retriever (Drive: ckpt/custom-bge-retriever-final)
        # + fine-tuned reranker (Drive: ckpt/custom-bge-reranker-final).
        # Used by the main route (Exp-2 / Exp-3).
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


CELL_20 = '''# 2.2 Dense retrieval dependencies (install faiss only when BGE/FAISS is needed)
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


CELL_22 = '''## A1: Retrieval ablation — BM25 / Custom BGE / Custom BGE + Reranker (Recall@K)

Three retrieval routes side-by-side, matching the "retrieval upgrade" experiment narrative:

1. **BM25** — used by Exp-1 baseline, feeds DeBERTa-v3-base
2. **Custom BGE** — fine-tuned dense retriever
3. **Custom BGE + Reranker** — final choice, feeds DeBERTa-v3-large / MoritzLaurer NLI

Cell A1.a computes the three Recall@K rows. Cell A1.c caches the BM25 top-200 candidates needed by the baseline experiment.
Custom BGE + Reranker candidates are produced dynamically by cell A2.b (no separate cache: only dev/test = 153×2 claims, fast enough).'''


CELL_23 = '''# A1.a: Retrieval evaluation — BM25 vs Custom BGE vs Custom BGE+Reranker (dev = 154 claims)
# Output: a table + cache/retrieval_eval_dev.pkl (idempotent)
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


# A1.a builds this single table only; it does NOT cache all-train candidates
# (those are produced by A1.c / A2.b to avoid duplicated storage).
KS = [3, 5, 10, 50, 100]
results = {}

# --- 1) BM25 alone (top-100 covers the entire K range) ---
print('[A1.a] BM25 top-100 ...')
t0 = time.time()
bm25_dev_cand = {cid: [d['id'] for d in get_top_200(c['claim_text'], n=100)]
                 for cid, c in tqdm(dev.items(), desc='BM25')}
print(f'  bm25 done in {time.time()-t0:.1f}s')
results['BM25'] = {k: recall_at_k(bm25_dev_cand, dev, k) for k in KS}

# --- 2) Custom BGE alone (lazy-load the fine-tuned retriever) ---
print('\\n[A1.a] Custom BGE top-100 ...')
try:
    from sentence_transformers import SentenceTransformer
    import faiss as _faiss
    custom_retriever_path = f'{CKPT}/custom-bge-retriever-final'
    if not os.path.isdir(custom_retriever_path):
        raise FileNotFoundError(f'{custom_retriever_path} missing — see README "Drive checkpoint sync"')
    _retr = SentenceTransformer(custom_retriever_path, device=DEVICE)
    _emb_path = f'{CACHE}/custom_bge_evidence_emb.npy'
    _idx_path = f'{CACHE}/custom_bge_faiss.index'
    _ids_path = f'{CACHE}/custom_bge_evi_ids.pkl'
    if os.path.exists(_emb_path) and os.path.exists(_idx_path) and os.path.exists(_ids_path):
        _idx = _faiss.read_index(_idx_path)
        _eids = load_pickle(_ids_path)
    else:
        print('  custom-bge corpus cache missing; skip Custom BGE evaluation (run cell A2.b first to build it)')
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

# --- 3) Custom BGE + Reranker (reuses final_dev_evidence from cell A2.b; optional) ---
if 'final_dev_evidence' in globals() and final_dev_evidence is not None and RETRIEVAL_SOURCE == 'custom_end_to_end':
    print('\\n[A1.a] Custom BGE + Reranker (reusing final_dev_evidence from A2.b) ...')
    results['Custom BGE + Reranker'] = {k: recall_at_k(final_dev_evidence, dev, k)
                                          for k in KS if k <= FINAL_TOP_K + 5}
else:
    print('\\n[A1.a] Reranker row requires cell A2.b to have run with RETRIEVAL_SOURCE=custom_end_to_end')

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


CELL_24 = '''# A1.c: Cache BM25 single-route top-200 candidates — required by the baseline experiment
import os, time
from tqdm.auto import tqdm


def cache_bm25_topk(claims, name, k=200, force_rebuild=False):
    path = f'{CACHE}/bm25_{name}.pkl'
    cached = load_candidate_cache(path, claims, name=f'bm25_{name}', max_len=k,
                                  known_evidence=evidence_filtered, force=force_rebuild)
    if cached is not None:
        return cached
    t0 = time.time()
    out = {}
    for cid, c in tqdm(claims.items(), desc=f'BM25 {name}'):
        out[cid] = [d['id'] for d in get_top_200(c['claim_text'], n=k)]
    save_pickle_atomic(out, path)
    print(f'  saved {path}  ({len(out)} claims, {time.time()-t0:.1f}s)')
    return out


bm25_train = cache_bm25_topk(train, 'train', k=200)
bm25_dev   = cache_bm25_topk(dev,   'dev',   k=200)
bm25_test  = cache_bm25_topk(test,  'test',  k=200)
print('BM25 candidate caches ready: bm25_train / bm25_dev / bm25_test')
'''


CELL_34 = '''# A2.b Final evidence selection (the retrieval -> classifier interface)
#
# Retrieval route selector (must match the chosen downstream classifier):
#   'bm25'              -> Exp-1 baseline (feeds DeBERTa-base)
#   'custom_end_to_end' -> Exp-2 large / Exp-3 NLI (main route)
#
# Output: final_{train,dev,test}_evidence: Dict[claim_id, List[evidence_id]]
# These dicts are the sole evidence source used by A3 training and A4 inference.

import numpy as np
import faiss
from tqdm.auto import tqdm
from sentence_transformers import CrossEncoder, SentenceTransformer

# ==========================================
# --- 1. MODEL SETUP (only when custom_end_to_end is selected) ---
# ==========================================

RETRIEVAL_SOURCE = 'custom_end_to_end'   # 'bm25' | 'custom_end_to_end'
FINAL_TOP_K = 5
FORCE_REBUILD_CUSTOM_BGE = False         # flip True to invalidate the corpus cache

# All three splits go through custom_end_to_end. Train (~11k claims) two-stage
# retrieval takes 30-60 minutes; for a quick check temporarily set ('dev', 'test').
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
        f'missing {custom_retriever_path}\\n'
        f'Please sync custom-bge-retriever-final/ from Colab Drive into ckpt/ (see README)'
    )
    assert os.path.isdir(custom_reranker_path), (
        f'missing {custom_reranker_path}\\n'
        f'Please sync custom-bge-reranker-final/ from Colab Drive into ckpt/ (see README)'
    )

    print('Loading Stage 2: FINE-TUNED BGE-Reranker...')
    rerank_model = CrossEncoder(custom_reranker_path, device=DEVICE)
    print('Loading Stage 1: FINE-TUNED BGE-Retriever...')
    retriever_model = SentenceTransformer(custom_retriever_path, device=DEVICE)

    print('\\n--- Initializing FAISS database (cache-first) ---')
    custom_faiss_index, evidence_ids = build_custom_bge_index(
        retriever_model, evidence_filtered, CACHE,
        force_rebuild=FORCE_REBUILD_CUSTOM_BGE,
    )

    print(f'\\n--- Stage 1+2 active splits: {RUN_SPLITS} ---')
    for split in ('train', 'dev', 'test'):
        if split not in RUN_SPLITS:
            print(f'  skipping split={split!r} (not in RUN_SPLITS)')
            continue
        print(f'\\n[{split}] Stage 1: wide-net (top 50)')
        stage1 = generate_stage1_cache(_SPLIT_DATA[split], retriever_model, custom_faiss_index, evidence_ids, top_k=50)
        print(f'[{split}] Stage 2: rerank (top {FINAL_TOP_K})')
        reranked = apply_dynamic_reranking(stage1, _SPLIT_DATA[split], FINAL_TOP_K, max_candidates=50, desc=f'Reranking {split}')
        globals()[f'final_{split}_evidence'] = reranked

elif RETRIEVAL_SOURCE == 'bm25':
    # Baseline path: top-5 from the BM25 cache (train/dev/test all need bm25_*.pkl)
    for split in ('train', 'dev', 'test'):
        cache_path = f'{CACHE}/bm25_{split}.pkl'
        assert os.path.exists(cache_path), f'missing {cache_path} — run cell A1.c first'
        globals()[f'final_{split}_evidence'] = load_pickle(cache_path)

else:
    raise ValueError(
        f'Unknown RETRIEVAL_SOURCE: {RETRIEVAL_SOURCE!r} '
        f'(allowed: "bm25" / "custom_end_to_end")'
    )

FINAL_EVIDENCE_SOURCE = RETRIEVAL_SOURCE
TOP_K = FINAL_TOP_K


# --- 3. Evaluation ---
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


CELL_35 = '''## A3: Three-backbone comparison — base / large / nli + two-stage fine-tuning

**Input format**: `[CLS] claim [SEP] evi_1 [SEP] evi_2 ... [SEP]`, attention-mask weighted mean-pooling, followed by a freshly initialised 4-way classification head (we do not reuse the backbone's original NLI head).

**Three routes** (switched via `CLS_BACKBONE_CHOICE` in cell A3.1):

| Route | Backbone | LoRA | Retrieval | Checkpoint |
|---|---|---|---|---|
| `'base'` | `microsoft/deberta-v3-base` (180M) | no (full fine-tune) | BM25 top-5 | `cls_best_base.pt` |
| `'large'` | `microsoft/deberta-v3-large` (435M) | r=16, α=32 | custom BGE+rerank top-5 | `cls_best_large.pt` |
| `'nli'` (final) | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (435M) | r=16, α=32 | custom BGE+rerank top-5 | `cls_best_nli.pt` |

The NLI backbone has been pre-fine-tuned on MNLI / FEVER / ANLI / LingNLI / WANLI, giving it the strongest starting point for fact-checking (cell A6 quantifies the gain). LoRA targets `query_proj` and `value_proj`; only the LoRA delta and the new classification head are trainable, which keeps the model within the Colab T4 VRAM budget.

**Two-stage training** (run for every route):
1. **Stage 1 — gold supervision**: train with gold evidence to warm up backbone + head.
2. **Stage 2 — retrieved supervision**: load Stage 1, continue with `final_train_evidence` (whichever retrieval route is active) to close the train-eval distribution gap.

Loss: class-weighted cross-entropy (inverse frequency, sum=4, up-weights DISPUTED) + label smoothing 0.1. Optimiser: AdamW with cosine warm-up. fp16 on CUDA. Best checkpoint is selected by dev accuracy.'''


CELL_36 = '''# A3.1: Label mapping + class weights + three-backbone configuration
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

# Inverse-frequency weights normalised so they sum to num_classes (keeps the loss magnitude stable)
inv = [1.0 / max(label_counts.get(l, 1), 1) for l in LABELS]
s = sum(inv)
CLASS_WEIGHTS = torch.tensor([w / s * len(LABELS) for w in inv], dtype=torch.float)
print(f'\\nClass weights (sum=4): {[f"{w:.2f}" for w in CLASS_WEIGHTS.tolist()]}')


# ----------------------------------------------------------------------
# Three-route configuration (change this single line to switch experiments)
# ----------------------------------------------------------------------
# 'base'  -> Exp-1 Baseline       : BM25 + microsoft/deberta-v3-base (full FT)
# 'large' -> Exp-2 Classifier Up  : custom BGE+rerank + microsoft/deberta-v3-large + LoRA
# 'nli'   -> Exp-3 FINAL          : custom BGE+rerank + MoritzLaurer NLI + LoRA
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
assert CLS_BACKBONE_CHOICE in BACKBONE_CONFIG, f'unknown CLS_BACKBONE_CHOICE={CLS_BACKBONE_CHOICE!r}'
CFG = BACKBONE_CONFIG[CLS_BACKBONE_CHOICE]
print(f'\\n[A3.1] CLS_BACKBONE_CHOICE = {CLS_BACKBONE_CHOICE!r}')
print(f'        backbone  = {CFG["name"]}')
print(f'        retrieval = {CFG["retrieval"]}')
print(f'        LoRA      = {CFG["use_lora"]}  max_len={CFG["max_length"]}  bs={CFG["batch_size"]}  ga={CFG["grad_accum"]}')
'''


CELL_37 = '''# A3.2: OOP classes — ClaimDS / BaselineClassifier / Trainer
# All three backbones share the same class hierarchy; per-route differences are
# driven by the CFG dictionary defined in cell A3.1:
#   - base : full fine-tuning, no LoRA
#   - large / nli : LoRA r=16, α=32 (target=query_proj/value_proj)
# In every route the classification head is a freshly initialised nn.Linear(h, 4).
# Even on the MoritzLaurer 3-class NLI backbone, the original NLI head is NOT
# reused — we always train a new 4-class head from scratch.

import gc
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model

CLS_BACKBONE = CFG['name']

class ClaimDS(Dataset):
    """Builds [CLS] claim [SEP] evi_1 [SEP] evi_2 ... [SEP] for the classifier.

    use_gold=True  : early training — evidence comes from c['evidences'] (gold)
    use_gold=False : later training / inference — evidence comes from evi_dict[cid][:top_k]
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
    """DeBERTa backbone + single linear classification head; large/nli use LoRA, base does full FT."""
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
        self.cls = nn.Linear(h, n_classes)   # fresh 4-class head (does NOT reuse the backbone's NLI head)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1)   # mean pooling
        return self.cls(self.dropout(pooled))


class Trainer:
    """fp16 (CUDA only) + class-weighted CE + label smoothing + cosine LR + best-ckpt by dev_acc."""
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

        # The optimiser only tracks parameters with requires_grad=True
        # (LoRA + cls head; or all parameters when LoRA is disabled).
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
                print(f'    saved -> {self.ckpt_path}')
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


CELL_38 = '''# A3 training entry — picks backbone / ckpt / training data source from CFG.
# Before training we re-check that RETRIEVAL_SOURCE matches CFG['retrieval'], to
# avoid e.g. running the baseline route on top of custom retrieval evidence.

CLS_BEST_PATH  = f'{CKPT}/{CFG["ckpt_name"]}'
FINAL_CKPT_PATH = CLS_BEST_PATH
CLS_FINAL_PATH = FINAL_CKPT_PATH

# Free retrieval-only large objects before classifier training (each route runs once)
release_large_objects(
    'bm25', 'evidence', 'bge_index_obj', 'bge_embeddings', 'bge_corpus_emb',
)

# Consistency check: the retrieval source used for training must match the one
# declared in BACKBONE_CONFIG for this choice.
assert RETRIEVAL_SOURCE == CFG['retrieval'], (
    f"RETRIEVAL_SOURCE={RETRIEVAL_SOURCE!r} does not match BACKBONE_CONFIG[{CLS_BACKBONE_CHOICE!r}]"
    f"['retrieval']={CFG['retrieval']!r} — go back to cell A2.b, fix RETRIEVAL_SOURCE, and rerun"
)
assert final_train_evidence is not None, (
    'final_train_evidence is None — go back to cell A2.b and include train in RUN_SPLITS'
)
assert final_dev_evidence is not None, 'final_dev_evidence is None — same as above'

print(f'[A3 training] choice={CLS_BACKBONE_CHOICE!r}  ckpt={CLS_BEST_PATH}')
print(f'              backbone={CFG["name"]}')
print(f'              train evidence source = {RETRIEVAL_SOURCE}')

# Two-stage training:
#   Stage 1 (gold supervision)      — evidence from c['evidences']; trains head + encoder
#   Stage 2 (retrieved supervision) — evidence from final_*_evidence; adapts to retrieval noise
RUN_STAGE_2 = True   # set False to keep only Stage 1 (faster, slightly lower dev_acc)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE = {DEVICE}')

# Stage 1: gold-supervised
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


CELL_40 = '''## A4: End-to-end prediction + three-way comparison

After training in A3, the classifier is paired with retrieval Top-K to produce dev / test predictions.

**Three comparison routes** (toggled by `CLS_BACKBONE_CHOICE` in cell A3.1, then Run All):

| Route | Retrieval | Classifier | Output |
|---|---|---|---|
| Exp-1 baseline | BM25 top-5 | DeBERTa-v3-base | `baseline-{dev,test}-predictions.json` |
| Exp-2 large | custom BGE+rerank top-5 | DeBERTa-v3-large + LoRA | `large-{dev,test}-predictions.json` |
| **Exp-3 NLI (final)** | custom BGE+rerank top-5 | MoritzLaurer NLI + LoRA | `nli-{dev,test}-predictions.json` <br>+ A4.3 per-evi: `nli-per-evi-{dev,test}-predictions.json` <br>+ A4.5 hybrid routing: `hybrid-{dev,test}-predictions.json` |

A4.1 is concatenated-evidence inference (one forward pass per claim, sees 5 evidence passages at once) and runs for every route.
A4.3 / A4.5 only activate on the NLI route (per-evidence NLI + routing are tied to the NLI pre-training).
Cell A6 reads the three prediction sets and renders the comparison table.'''


CELL_41 = '''# A4.1: End-to-end prediction + save dev/test predictions
# Output filenames are driven by CFG["pred_prefix"]: baseline- / large- / nli-
import json, torch
from transformers import AutoTokenizer
from tqdm.auto import tqdm

# Load the checkpoint that corresponds to the current CFG
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
            evis = [next(iter(evidence_filtered))]   # eval.py requires at least 1 evidence id
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


CELL_42 = '''# A4.2: Invoke the official eval.py to compute Hmean = 2FA / (F + A)
import subprocess

cmd = [
    'python', f'{DATA}/eval.py',
    '--predictions', dev_pred_path,
    '--groundtruth', f'{DATA}/dev-claims.json',
]
print(' '.join(cmd))
r = subprocess.run(cmd, capture_output=True, text=True)
print(r.stdout if r.returncode == 0 else r.stderr)
'''


CELL_43 = '''## A4.3: NLI-style per-evidence inference (only active on the `'nli'` route)

We re-use the NLI-pre-trained backbone as a per-evidence NLI model, as a counterpart to the concatenated-evidence inference in A4.1:

- For each evidence i we run `(claim, evi_i)` independently → 4-class softmax `p_i`
- Default prediction is `argmax(mean_i p_i)`
- Strength: stronger on SUPPORTS / REFUTES than the concatenated inference
- Weakness: when scoring a single evidence the model almost never emits NEI / DISPUTED, so overall accuracy is below A4.1

> We initially planned a "one evidence strongly supports + another strongly refutes -> DISPUTED" τ-rule.
> A τ sweep on dev shows that no threshold can reliably trigger it (per-evidence probabilities skew SUP/REF),
> so we leave τ=1.01 (rule disabled). DISPUTED is better fixed on the training side (over-sampling / focal loss).

The `base` and `large` routes skip this cell automatically. The output goes to `nli-per-evi-*-predictions.json`,
which is fused with A4.1's `nli-*-predictions.json` by **A4.5 hybrid routing** to produce the final prediction.'''


CELL_44 = '''# A4.3: per-evidence NLI inference + aggregation (NLI route only; base/large are skipped)
if CLS_BACKBONE_CHOICE != 'nli':
    print(f'[A4.3] skip per-evidence NLI inference (CHOICE={CLS_BACKBONE_CHOICE!r}, '
          f'only meaningful for the nli backbone)')
else:
    import json, numpy as np, torch
    from collections import Counter
    from transformers import AutoTokenizer
    from tqdm.auto import tqdm

    # Make sure we use the final NLI checkpoint
    model.load_state_dict(torch.load(FINAL_CKPT_PATH, map_location=DEVICE))
    model.eval()

    _NLI_TOK = AutoTokenizer.from_pretrained(CLS_BACKBONE, use_fast=False)
    SUP = LABEL2ID['SUPPORTS']
    REF = LABEL2ID['REFUTES']
    NEI = LABEL2ID['NOT_ENOUGH_INFO']
    DIS = LABEL2ID['DISPUTED']

    NLI_MAX_LEN = 320  # one evidence at a time — shorter than concatenated, saves VRAM

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
        print(f'  tau={tau:>4.2f}  dev_acc={acc:.4f}{marker}')
    print(f'\\nBest tau = {best_tau}  dev_acc = {best_acc:.4f}')

    # Output filenames use a "nli-per-evi-" prefix so they don't collide with A4.1
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

    print('\\n=== Dev confusion matrix (rows=gold, cols=pred) @ best tau ===')
    header = ' ' * 18 + '  '.join(f'{l[:5]:>5s}' for l in LABELS)
    print(header)
    for gl in LABELS:
        row = [best_cm.get((gl, pl), 0) for pl in LABELS]
        print(f'  {gl:<16s}' + '  '.join(f'{v:>5d}' for v in row))

    NLI_BEST_TAU = best_tau
'''


CELL_45 = '''## A4.5: Hybrid routing — the final scheme on the NLI route

A4.1 concatenated NLI and A4.3 per-evidence NLI are **complementary** across labels: A4.1 is conservative and reliable on NEI, while A4.3 is sharper on SUP/REF. Based on the dev confusion matrices (printed by this cell when run), we route as follows:

```
if concat_pred (A4.1) == 'NOT_ENOUGH_INFO':
    final = NEI            # trust A4.1's NEI judgement
else:
    final = per_evi_pred   # use A4.3 between SUPPORTS / REFUTES
```

The `base` and `large` routes skip this cell automatically. On the NLI route it produces `hybrid-{dev,test}-predictions.json` — **our final submission file**.

> Historical numbers (single run, indicative only — refresh from cell A6 after re-running):
> | Path | A | F | Hmean |
> |---|---|---|---|
> | A4.1 concat (NLI) | 0.5260 | 0.1057 | 0.1761 |
> | A4.3 NLI aggregated | 0.4675 | 0.1057 | 0.1725 |
> | **A4.5 Hybrid** | **0.5325** | **0.1057** | **0.1764** |'''


CELL_46 = '''# A4.5: Hybrid routing — baseline-NEI OR NLI-(SUP/REF)  (NLI route only)
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

    # Read predictions from A4.1 (nli-*-predictions.json) and A4.3 (nli-per-evi-*-predictions.json)
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

    # Compare three NLI-side paths on dev
    acc_b, cm_b, _ = _score(f'{ROOT}/nli-dev-predictions.json',         dev)
    acc_n, cm_n, _ = _score(f'{ROOT}/nli-per-evi-dev-predictions.json', dev)
    acc_h, cm_h, _ = _score(hybrid_dev_path,                            dev)
    print(f'\\nNLI cat (A4.1)     A = {acc_b:.4f}')
    print(f'NLI per-evi (A4.3) A = {acc_n:.4f}   delta vs cat = {acc_n-acc_b:+.4f}')
    print(f'Hybrid (A4.5)      A = {acc_h:.4f}   delta vs cat = {acc_h-acc_b:+.4f}')

    _print_cm(cm_h, '=== Hybrid confusion matrix ===')

    print('\\n--- eval.py: hybrid-dev ---')
    r = subprocess.run(['python', f'{DATA}/eval.py',
                        '--predictions', hybrid_dev_path,
                        '--groundtruth', f'{DATA}/dev-claims.json'],
                       capture_output=True, text=True)
    print(r.stdout if r.returncode == 0 else r.stderr)
'''


CELL_47 = '''## A5: End-to-end smoke test

Runs the full retrieve -> rerank -> classify -> eval pipeline on 8 dev claims in under 2 minutes, to confirm nothing is broken.
Prerequisites for this cell: cell A2.b has completed (so `retriever_model` / `rerank_model` / `custom_faiss_index` are in memory) and cell A4.1's `predict()` function is defined.'''


CELL_48 = '''# A5: end-to-end smoke test — 8 dev claims through the full pipeline
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

print('=== A5.1 file existence check ===')
all_ok = all(_check(g, ps) for g, ps in REQUIRED_PATHS.items())
if not all_ok:
    print('\\nKey files are missing — see the README section "Drive checkpoint sync" '
          'before re-running')
else:
    print('\\n=== A5.2 8 dev claims through the full pipeline ===')
    dev_mini = dict(list(dev.items())[:8])

    # Retrieval: stage-1 (50) + stage-2 (5) on the fly; no hybrid_*.pkl dependency
    mini_stage1 = generate_stage1_cache(dev_mini, retriever_model, custom_faiss_index, evidence_ids, top_k=50)
    mini_evi    = apply_dynamic_reranking(mini_stage1, dev_mini, top_k=FINAL_TOP_K, max_candidates=50, desc='A5 mini-rerank')

    # Classification (A4.1 concatenated)
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


CELL_49 = '''## A6: Three-way comparison table

Once you have run with `CLS_BACKBONE_CHOICE` set to `'base'`, `'large'`, and `'nli'` in turn, this cell reads the three dev prediction JSON files, calls eval.py, and renders a comparison table. A row with `–` means that experiment has not been run yet.'''


CELL_50 = '''# A6: read baseline-/large-/nli- predictions and produce the comparison table
import os, json, subprocess
from collections import OrderedDict

EXPERIMENTS = OrderedDict([
    ('Exp-1 Baseline (BM25 + DeBERTa-base)',          'baseline-dev-predictions.json'),
    ('Exp-2 + Retrieval upgrade + DeBERTa-large',     'large-dev-predictions.json'),
    ('Exp-3 Final: NLI backbone (A4.1 cat)',          'nli-dev-predictions.json'),
    ('Exp-3 Final: NLI per-evidence (A4.3)',          'nli-per-evi-dev-predictions.json'),
    ('Exp-3 Final: Hybrid routing (A4.5) [adopted]',  'hybrid-dev-predictions.json'),
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
print('\\n(Empty rows: set CLS_BACKBONE_CHOICE in cell A3.1 to the missing value and Run All.)')
'''


# ======================================================================
# Cells originally authored by the team — translate ONLY the comments
# (keep all executable code identical to the user's original)
# ======================================================================

# Cell 4 (id=92e06292) — 1.1 path & device config
CELL_4_TRANSLATE = {
    "# 1.1 路径与设备配置（团队协作：Colab 走 Drive shortcut，本地走相对路径）":
        "# 1.1 Path and device configuration (team workflow: Drive shortcut on Colab, relative path locally)",
    "# 设计：":
        "# Design:",
    "#   - Colab : ROOT = /content/drive/MyDrive/comp90042":
        "#   - Colab : ROOT = /content/drive/MyDrive/comp90042",
    "#             （团队成员对该文件夹做了 shortcut，所以每个人 MyDrive 下都能看到）":
        "#             (the team has set up a shortcut so every member sees this folder under MyDrive)",
    "#   - 本地  : ROOT = notebook 所在目录的绝对路径":
        "#   - Local : ROOT = absolute path of the directory containing the notebook",
    "#             （队友只要 clone 仓库，data/ 和 cache/ 平级即可，不用改代码）":
        "#             (after cloning the repo, data/ and cache/ live next to the notebook; no code changes needed)",
    "#   - 任何人都可以用 COMP90042_ROOT 环境变量覆盖，方便挪盘":
        "#   - Any user can override via the COMP90042_ROOT env var (useful when moving disks)",
    "# 1) 优先读环境变量（最高优先级，任何人想挪到 SSD/外接盘 export 一下就行）":
        "# 1) Environment variable takes precedence (highest priority; users moving to SSD/external disk just export it)",
    "# 2) Colab：挂 Drive 后用团队约定的 shortcut 路径":
        "# 2) Colab: after mounting Drive, use the team's shortcut path",
}


# Cell 6 (id=02c00d02) — 1.2 NLTK resources
CELL_6_TRANSLATE = {
    "# 1.2 NLTK 资源（如果是 Colab 第一次跑，取消下面 pip 注释）":
        "# 1.2 NLTK resources (uncomment the pip line below on a fresh Colab session)",
}


# Cell 8 (id=af2021d0) — 1.3 load data
CELL_8_TRANSLATE = {
    "# 1.3 读取数据": "# 1.3 Load data",
}


# Cell 9 (id=fe653981) — 1.4 schema sample
CELL_9_TRANSLATE = {
    "# 1.4 抽样查看一条 train 数据，确认 schema":
        "# 1.4 Sample one training claim and inspect its schema",
}


# Cell 10 (id=ecd3efa5) — 1.5 EDA markdown
CELL_10_TRANSLATE = {
    "## 1.5 EDA：长度分布、标签分布、evidence 数":
        "## 1.5 EDA: length distribution, label distribution, evidence count",
    "四张图一次出，写报告时直接引用：":
        "Four sub-plots in one figure for direct reuse in the report:",
    "| 子图 | 用途 |":
        "| Sub-plot | Purpose |",
    "| Claim 长度 | 决定 tokenizer max_length（claim 端） |":
        "| Claim length | Informs tokenizer max_length on the claim side |",
    "| 4 类标签分布 | 论证 class-weighting 必要性，DISPUTED 严重不平衡 |":
        "| 4-way label distribution | Motivates class-weighting; DISPUTED is severely imbalanced |",
    "| Evidence 条数 | 决定 top-k 上界（5 是 dev/train 上的 95-percentile）|":
        "| Evidence count per claim | Informs the top-k upper bound (5 is the 95-percentile on dev/train) |",
    "| Evidence 词长 | 决定 max_length（evidence 端，配合 SEP 拼接 5 条评估）|":
        "| Evidence token length | Informs max_length on the evidence side (used when concatenating 5 evidences with [SEP]) |",
}


# Cell 11 (id=76249762) — 1.5 EDA code
CELL_11_TRANSLATE = {
    "# 1.5 EDA: 4 张子图": "# 1.5 EDA: 4-panel figure",
    "# 排序，避免出现混合顺序": "# Sort to avoid mixed orderings",
    "# (1) Claim 长度（词数）": "# (1) Claim length (word count)",
    "# (2) Label 分布": "# (2) Label distribution",
    "# (3) Evidence 条数": "# (3) Evidence count per claim",
    "# (4) Evidence 词长": "# (4) Evidence token length",
    "# 保存到 cache/ 方便报告里直接引用":
        "# Save to cache/ for direct embedding in the report",
}


# Cell 12 (id=ee931891) — 1.6 evidence filter
CELL_12_TRANSLATE = {
    "# 1.6 evidence filtering cache（默认加载 cache/evidence_filtered.pkl）":
        "# 1.6 Evidence-filtering cache (loads cache/evidence_filtered.pkl by default)",
}


# Cell 18 (id=1140a37f) — 2.1 BM25
CELL_18_TRANSLATE = {
    "# 2.1 BM25 cache（默认加载 cache/bm25.pkl；缺失或无效才重建）":
        "# 2.1 BM25 cache (loads cache/bm25.pkl by default; rebuilds only when missing or invalid)",
}


# Cell 20 already rewritten above (CELL_20).

# Cell 21 (id=e9f11746) — BGE encoding lazy init
CELL_21_TRANSLATE = {
    "# 2.x BGE / FAISS dense retrieval（lazy init；只有缺 cache 或强制重建时才加载）":
        "# 2.x BGE / FAISS dense retrieval (lazy init; loaded only when the cache is missing or a force rebuild is requested)",
}


# ======================================================================
# Driver: apply translations
# ======================================================================

REWRITES = [
    ('e0c87bda', CELL_5),
    ('14267b8a', CELL_7),
    ('6deed960', CELL_20),
    ('23320e3f', CELL_22),
    ('ee509c39', CELL_23),
    ('06c73a5d', CELL_24),
    ('e50ff75d', CELL_34),
    ('430ec164', CELL_35),
    ('14b2d465', CELL_36),
    ('ba911328', CELL_37),
    ('bIGHVIH4wcXn', CELL_38),
    ('c95d8f69', CELL_40),
    ('ac06df41', CELL_41),
    ('3b2f2f28', CELL_42),
    ('711f85a7', CELL_43),
    ('bea07c65', CELL_44),
    ('371ae36a', CELL_45),
    ('7f9110d6', CELL_46),
]

# After the 4 new inserted cells (A5 markdown + code, A6 markdown + code), the
# inserted cells use auto-generated IDs (uuid hex). To target them, identify by
# the first line of source instead of cell_id.
INSERTED_FIRST_LINES = {
    '## A5: 端到端 smoke test': CELL_47,
    "# A5: end-to-end smoke test —— 8 条 dev claim 验证整条 pipeline": CELL_48,
    '## A6: 三路对照实验汇总表': CELL_49,
    "# A6: 读 baseline-/large-/nli- 三套 dev predictions": CELL_50,
}


# Cells where we only patch specific Chinese phrases (the user authored the rest)
LINE_LEVEL = [
    ('92e06292', CELL_4_TRANSLATE),
    ('02c00d02', CELL_6_TRANSLATE),
    ('af2021d0', CELL_8_TRANSLATE),
    ('fe653981', CELL_9_TRANSLATE),
    ('ecd3efa5', CELL_10_TRANSLATE),
    ('76249762', CELL_11_TRANSLATE),
    ('ee931891', CELL_12_TRANSLATE),
    ('1140a37f', CELL_18_TRANSLATE),
    ('e9f11746', CELL_21_TRANSLATE),
]


def patch_chinese_phrases(cells, cell_id, mapping):
    for c in cells:
        if c.get('id') == cell_id:
            src = ''.join(c['source'])
            for zh, en in mapping.items():
                src = src.replace(zh, en)
            c['source'] = src.splitlines(keepends=True)
            if c['source'] and c['source'][-1].endswith('\n'):
                c['source'][-1] = c['source'][-1].rstrip('\n')
            return
    raise SystemExit(f'cell {cell_id} not found (line-level patch)')


def patch_by_first_line(cells, first_line_starts, new_src):
    for c in cells:
        src = ''.join(c['source'])
        first_line = next((l for l in src.splitlines() if l.strip()), '')
        if first_line.startswith(first_line_starts):
            c['source'] = new_src.splitlines(keepends=True)
            if c['source'] and c['source'][-1].endswith('\n'):
                c['source'][-1] = c['source'][-1].rstrip('\n')
            return True
    return False


def main():
    with open(NB) as f:
        nb = json.load(f)

    # Full rewrites of cells with stable IDs
    for cid, new_src in REWRITES:
        replace_cell(nb['cells'], cid, new_src)
        print(f'  rewrote cell id={cid}')

    # Newly inserted cells — identify by first line
    for first_line, new_src in INSERTED_FIRST_LINES.items():
        ok = patch_by_first_line(nb['cells'], first_line, new_src)
        print(f'  rewrote inserted cell starting with {first_line!r:60s} -> {"OK" if ok else "NOT FOUND"}')

    # Line-level phrase replacements (preserve user code)
    for cid, mapping in LINE_LEVEL:
        patch_chinese_phrases(nb['cells'], cid, mapping)
        print(f'  patched comments in cell id={cid}  ({len(mapping)} phrases)')

    with open(NB, 'w') as f:
        json.dump(nb, f, indent=2)
    print('\\nDone.')


if __name__ == '__main__':
    main()
