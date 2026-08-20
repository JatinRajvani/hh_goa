# 03 — Dataset & Ingestion

This document explains our optimized data solution for handling the large-scale Hugging Face dataset within hardware and hosting constraints.

---

## 📊 Dataset: `ai4bharat/MSMARCO-XI`
*   **Total Size**: **55.6 GB** (Uncompressed parquet files).
*   **Languages**: 14 (English and 13 native Indic languages).
*   **Characteristics**: Short passages (averaging 50–100 words) paired with queries in native scripts, mapping to MSMARCO identifiers.

---

## 🚫 The 55.6 GB Memory Constraint
Attempting to download, compile, or load a 55.6 GB corpus on standard developer laptops (16GB RAM) or a free hosting tier (Render Free: 512MB RAM) results in instant Out-of-Memory (OOM) failures.

### Our Solution: Streaming & Capping
1.  **Parquet Streaming**: Instead of downloading the full dataset, we stream the parquet dataset splits directly from Hugging Face using the `datasets` streaming API.
2.  **Capped Corpus**: We ingest and process exactly **1,500 unique passages per language**.
3.  **Total Context Size**: With 14 languages supported, this results in a localized, production-ready index of **21,000 total passages** (14 × 1,500).

---

## 🔄 Ingestion & Serialization Pipeline
The ingestion process is governed by the script `ingestion/build_multilingual_indices.py`:

```python
# Ingestion Logic Overview
for lang_code in LANGUAGES:
    # 1. Stream the parquet split from Hugging Face
    dataset = load_dataset("ai4bharat/MSMARCO-XI", lang_code, split="train", streaming=True)
    
    # 2. Extract up to 1,500 unique passages
    passages = {}
    for record in dataset:
        passages[passage_id] = passage_text
        if len(passages) >= 1,500:
            break
```

### Generated Files:
*   **JSON Map**: A static index mapping file `data/index/id_mapping_{lang}.json` containing:
    *   `ordered_ids`: An ordered array of passage IDs for flat indexing.
    *   `mapping`: A key-value object linking passage IDs to their text snippets.
*   **FAISS Vector Binaries**: Flat index files `data/index/index_{lang}.faiss` containing pre-computed 384-dimensional normalized vectors (if running dev environment).

---

## 🚀 Deployment Optimization
In Render production, we exclude the heavy FAISS binary index files from deployment using `.gitignore`. The cloud backend only loads the lightweight JSON index mappings, allowing it to perform fast token-scans (BM25) with a memory footprint of **under 45MB RAM** (preserving 90% overhead).
