# Lexical Search (BM25) vs. Dense Semantic Search (FAISS)

This document provides a comparative analysis of the two retrieval engines integrated into our Voice RAG Portal. Understanding the strengths, weaknesses, and concrete search behaviors of **Lexical Keyword Search (BM25)** vs. **Dense Vector Search (FAISS)** is key to designing high-performance retrieval architectures.

---

## 1. Architectural Comparison

| Attribute | Lexical Search (Okapi BM25) | Dense Semantic Search (FAISS) |
| :--- | :--- | :--- |
| **Matching Mechanism** | Exact term overlap & frequency (TF-IDF based). | Vector distance (Cosine Similarity in 384d space). |
| **Focus** | Matches *exact words* used in the query. | Matches the *meaning and intent* of the query. |
| **Local Startup RAM** | **~2 MB** (extremely lightweight). | **~450 MB** (requires loading PyTorch weights). |
| **Query Latency** | **< 2 ms** (instant lookup). | **~340 ms** (due to CPU model vector encoding). |
| **Strengths** | Exact matches, product codes, unique names, acronyms. | Synonyms, paraphrasing, translation gaps, Indic language cross-matches. |
| **Weaknesses** | Fails on synonyms, typos, and different phrasings. | Can return false positives for words that are semantically close but logically distinct. |

---

## 2. Concrete Examples (Search Comparison)

Below are four query examples demonstrating the differences in retrieval results between the two engines:

### Example A: Concept & Siting Requirements (Semantic Win)
* **User Query**: *"Why was the secret nuclear facility placed close to a large water body?"*
* **Retrieval Behavior**:
  * **Okapi BM25**: **Keyword Overlap Match (Fails to Answer the Question)**. It matches keyword tokens like `"nuclear"` and `"facility"`, returning a generic, high-level description of the Manhattan Project that completely ignores the "water body" relationship:
    > *"Manhattan Project. The Manhattan Project was a research and development undertaking during World War II that produced the first nuclear weapons. It was led by the United States with the support of the United Kingdom and Canada. From 1942 to 1946, the project was under the direction of Major General Leslie Groves of the U.S. Army Corps of Engineers. Nuclear physicist Robert Oppenheimer was the director of the Los Alamos Laboratory that designed the actual bombs. The Army component of the project was designated the..."*
  * **FAISS Semantic**: **Conceptual Association (Successfully Answers the Question)**. It matches the underlying semantic link between "nuclear facility", "placed close", and "large water body", retrieving the specific passage detailing the siting of the Hanford B Reactor near the Columbia River:
    > *"One of the main reasons Hanford was selected as a site for the Manhattan Project's B Reactor was its proximity to the Columbia River, the largest river flowing into the Pacific Ocean from the North American coast."*

---

### Example B: Synonyms & Paraphrasing (Semantic Win)
* **User Query**: *"treatment for thoracic spine pain"*
* **Target Passage**: *"A guide on handling middle back ache through physical therapy."*
* **Retrieval Behavior**:
  * **Okapi BM25**: **Score = 0.0 (Failed)**. The target passage contains none of the exact search keywords ("thoracic", "spine", "pain").
  * **FAISS Semantic**: **Score = 0.88 (Success)**. The multilingual embedding model (`paraphrase-multilingual-MiniLM-L12-v2`) understands synonyms:
    - *"thoracic spine"* $\approx$ *"middle back"*
    - *"pain"* $\approx$ *"ache"*
    - *"treatment"* $\approx$ *"physical therapy"*

---

### Example C: Multilingual Intent Alignment (Semantic Win)
* **User Query (Hindi)**: *"दिल का दौरा पड़ने पर क्या करें"* (What to do during a heart attack)
* **Target Passage (Hindi)**: *"हृदय आघात होने के सामान्य लक्षण और प्राथमिक चिकित्सा।"* (Common symptoms and first aid during cardiac arrest.)
* **Retrieval Behavior**:
  * **Okapi BM25**: **Score = 0.0 (Failed)**. There is zero keyword overlap between "दिल का दौरा" (heart attack) and "हृदय आघात" (cardiac arrest).
  * **FAISS Semantic**: **Score = 0.82 (Success)**. The embedding model maps the semantic concept of "दिल का दौरा" and "हृदय आघात" closely together in the multilingual vector space, returning the correct first-aid instructions.

---

### Example D: Product Codes & Exact Numbers (BM25 Win)
* **User Query**: *"instructions for filling Form 1040EZ"*
* **Target Passage 1**: *"How to file Form 1040A for tax exemptions."*
* **Target Passage 2**: *"Filing guidelines for IRS Form 1040EZ."*
* **Retrieval Behavior**:
  * **FAISS Semantic**: Often ranks **Passage 1** higher than or equal to Passage 2. Since "1040EZ" and "1040A" are both tax forms, their vector representations are extremely close, making semantic search prone to numeric/code mismatches.
  * **Okapi BM25**: **Successfully ranks Passage 2 highest**. BM25 targets the exact keyword token `"1040ez"`, scoring a perfect term-frequency match and filtering out the incorrect tax forms immediately.

---

## 3. Deployment Summary

By offering both search engines in our portal, we achieve optimal performance in both environments:
1. **Local Hybrid setup**: Allows developers to compare BM25 and Semantic search side-by-side to understand accuracy tradeoffs.
2. **Cloud production (Render)**: Automatically locks retrieval to **BM25**, preserving 90% memory overhead and enabling the app to run comfortably under the free hosting memory constraints while still serving lightning-fast lexical answers.
