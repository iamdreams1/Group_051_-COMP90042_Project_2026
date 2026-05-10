# Cache-First Execution Plan for Colab

This document is the required execution standard for this project. Future notebook or code changes must follow this plan.

## 1. Goal

The project must run on the free version of Google Colab without exhausting the default CPU RAM limit.

The default behavior must be cache-first:

- Load existing cache whenever possible.
- Rebuild cache only when it is missing, corrupted, incompatible, or explicitly forced.
- Avoid keeping multiple large retrieval objects in memory at the same time.
- Make every expensive step resumable or reusable by the whole team.

## 2. Required Runtime Defaults

All code must use the shared project root:

```python
ROOT = os.environ.get("COMP90042_ROOT", "/content/drive/MyDrive/comp90042")
DATA = os.path.join(ROOT, "data")
CACHE = os.path.join(ROOT, "cache")
CKPT = os.path.join(ROOT, "ckpt")
```

All expensive rebuilds must be controlled by one dictionary:

```python
FORCE_REBUILD = {
    "filtered": False,
    "bm25": False,
    "bge": False,
    "eval": False,
    "hybrid": False,
    "rerank": False,
}
```

No code cell should rebuild a heavy artifact unless the matching flag is `True` or the cache check fails.

## 3. Required Cache Files

The following cache files are part of the official workflow.

| File | Purpose | Git Policy |
|---|---|---|
| `cache/evidence.pkl` | Faster loading of raw evidence | Drive only |
| `cache/evidence_filtered.pkl` | Filtered evidence dict and ordered ids | Drive only |
| `cache/bm25.pkl` | BM25Okapi index and evidence id order | Drive only |
| `cache/bge_evidence_emb.npy` | BGE evidence embeddings | Drive only |
| `cache/bge_faiss.index` | FAISS dense retrieval index | Drive only |
| `cache/retrieval_eval_dev.pkl` | A1 Recall@K results | Drive or local cache |
| `cache/hybrid_train.pkl` | Hybrid candidates for train | May be committed if small |
| `cache/hybrid_dev.pkl` | Hybrid candidates for dev | May be committed if small |
| `cache/hybrid_test.pkl` | Hybrid candidates for test | May be committed if small |
| `cache/reranked_train.pkl` | Reranked train candidates | Drive or local cache |
| `cache/reranked_dev.pkl` | Reranked dev candidates | Drive or local cache |
| `cache/reranked_test.pkl` | Reranked test candidates | Drive or local cache |

## 4. Mandatory Cache Rules

All cache-producing code must follow these rules:

1. Check whether the cache exists before doing any heavy computation.
2. Validate loaded cache before using it.
3. Rebuild only if validation fails or `FORCE_REBUILD[...]` is enabled.
4. Save with atomic writes:
   - write to `path + ".tmp"` first
   - then replace with `os.replace(tmp, path)`
5. After building large intermediate objects, immediately release them with `del` and `gc.collect()`.

## 5. Required Validation

Each cache must be checked before use.

Hybrid and reranked caches must validate:

- claim ids exactly match the split
- no missing claims
- no extra claims
- values are lists of evidence ids
- each list has no duplicate evidence ids
- hybrid candidate length is at most 400

BM25 cache must validate:

- `bm25.corpus_size == len(evi_ids_filtered)`
- cached evidence id order matches the current filtered evidence id order
- tokenizer/filter configuration matches the current notebook version

BGE/FAISS cache must validate:

- FAISS index exists before loading
- `index.ntotal == len(evi_ids_filtered)`
- embedding dimension matches the model setting
- embedding file is not loaded fully into memory unless needed

## 6. Execution Order

The notebook must follow this order:

1. Configure paths, device, and dependencies.
2. Define cache helper functions.
3. Load train/dev/test claims.
4. Load `evidence.pkl`; create it only if missing.
5. Load `evidence_filtered.pkl`; create it only if missing.
6. Load `bm25.pkl`; build it only if missing or invalid.
7. If all `hybrid_*.pkl` files already exist, skip BGE model loading and FAISS initialization.
8. If any hybrid cache is missing, load/build BGE and FAISS only then.
9. Load `retrieval_eval_dev.pkl`; recompute A1.a only if missing or forced.
10. Load or build `hybrid_{train,dev,test}.pkl`.
11. Load or build `reranked_{train,dev,test}.pkl`.
12. Run classifier training or prediction.
13. Release retrieval-only objects before classifier training when possible.

## 7. Explicitly Forbidden Patterns

Do not add code that does any of the following by default:

```python
tokenized_corpus = [nltk_tokenize(evidence_filtered[eid]) for eid in evi_ids_filtered]
bm25 = BM25Okapi(tokenized_corpus)
```

unless it is inside a cache-aware function that first checks `cache/bm25.pkl`.

Do not run full BGE encoding unless `cache/bge_evidence_emb.npy` is missing or `FORCE_REBUILD["bge"] = True`.

Do not rerun A1.a dev retrieval every time. It must load `cache/retrieval_eval_dev.pkl` by default.

Do not load CrossEncoder unless at least one `reranked_*.pkl` file is missing or invalid.

## 8. Colab RAM Requirements

To stay within free Colab RAM limits:

- Delete raw `evidence` after `evidence_filtered` is loaded or built.
- Delete `tokenized_corpus` immediately after BM25 is built.
- Use `np.load(..., mmap_mode="r")` for large embedding files.
- Do not keep full embeddings in RAM if FAISS index is already available.
- Do not keep BM25, BGE model, FAISS index, CrossEncoder, and classifier model alive together unless strictly needed.
- Run `gc.collect()` after releasing large retrieval objects.

## 9. Team Workflow

Before running the notebook, every team member must check:

- Google Drive shortcut points to `/content/drive/MyDrive/comp90042`
- `data/` exists under `ROOT`
- `cache/` exists under `ROOT`
- required shared cache files are present if they want the fast path

Default team behavior:

```python
FORCE_REBUILD = {
    "filtered": False,
    "bm25": False,
    "bge": False,
    "eval": False,
    "hybrid": False,
    "rerank": False,
}
```

Only change one flag to `True` when intentionally rebuilding that artifact.

## 10. Acceptance Criteria

A correct implementation of this plan must satisfy:

- Running the notebook with all cache files present does not rebuild BM25.
- Running the notebook with all cache files present does not rerun A1.a retrieval.
- Running the notebook with all hybrid cache files present does not load BGE or FAISS.
- Deleting only `cache/hybrid_dev.pkl` rebuilds only dev hybrid candidates.
- Deleting only `cache/retrieval_eval_dev.pkl` recomputes only retrieval evaluation.
- Hybrid caches contain exactly the expected split claim ids.
- Each hybrid candidate list has at most 400 evidence ids.
- Colab free CPU RAM does not reach the 12GB limit during the cache-hit path.
