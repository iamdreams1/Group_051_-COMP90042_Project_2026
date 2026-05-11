# COMP90042 Group 051 Baseline

This branch contains the BM25 + DeBERTa baseline for the COMP90042 2026 project.

The current baseline intentionally does not use BGE, FAISS, RRF, hybrid retrieval, reranking, or any external dataset. The goal is to keep one reproducible baseline pipeline before further modelling work.

## System

1. BM25 retrieves candidate evidence passages for each claim.
2. The final classifier input uses BM25 Top-5 evidence passages.
3. `microsoft/deberta-v3-base` encodes `claim [SEP] evidence_1 ... evidence_5`.
4. A linear classification head predicts one of:
   `SUPPORTS`, `REFUTES`, `NOT_ENOUGH_INFO`, `DISPUTED`.
5. `eval.py` reports evidence F-score, claim accuracy, and harmonic mean on dev.

## Important Files

- `Group_051__COMP90042_Project_2026.ipynb`: main notebook, configured for BM25 baseline mode.
- `README.md`: this running note.

The notebook expects the project data and BM25 cache files to be available locally:

- `data/train-claims.json`, `data/dev-claims.json`, `data/test-claims-unlabelled.json`
- `data/dev-claims-baseline.json`
- `data/evidence.json`
- `cache/evidence.pkl`
- `cache/evidence_filtered.pkl`
- `cache/bm25.pkl`
- `cache/bm25_train.pkl`, `cache/bm25_dev.pkl`, `cache/bm25_test.pkl`

Large BGE/FAISS/hybrid cache files are not required for this branch.

## Setup

Recommended Python packages:

```bash
python -m pip install rank_bm25 sentencepiece protobuf tiktoken transformers==4.40.0 nltk pandas matplotlib scikit-learn torch tqdm
```

If running the notebook in Colab, run cells from the top. The notebook is configured with:

```python
RUN_HYBRID_EXPERIMENTS = False
RETRIEVAL_SOURCE = "bm25"
FINAL_TOP_K = 5
```

## Run The Baseline

Run the notebook from the top after placing the data/cache files above in the expected paths. The prediction cells write:

- `baseline-dev-predictions.json`
- `baseline-test-claims-predictions.json`
- `ckpt/cls_best.pt`

Evaluate dev predictions:

```bash
python eval.py --predictions baseline-dev-predictions.json --groundtruth data/dev-claims.json
```

## Current Caveat

The latest saved baseline predictions were generated before the conservative class-weight change. They showed a collapsed prediction distribution, so the baseline should be rerun before reporting final numbers.

The notebook and script now use inverse-frequency class weights without an extra `DISPUTED` booster.

## Git Notes

This branch is pushed with only the notebook and README changes. Large data/cache files are intentionally kept out of git because several exceed GitHub's normal 100 MB file limit.
