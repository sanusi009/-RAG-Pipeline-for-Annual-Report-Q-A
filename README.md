# 📊 RAG Pipeline for Annual Report Q&A

> End-to-end Retrieval-Augmented Generation on a corporate annual report PDF — with LLM evaluation, embedding drift monitoring, and an interactive Gradio dashboard.

---

## 🧠 What this project does

This notebook builds a production-style RAG pipeline that lets you ask natural-language questions against a company annual report and get **grounded, cited answers** — with automatic quality scoring and drift detection built in.

---

## 🗂️ Pipeline overview

```
PDF → Extract text → Two-stage chunking → Embed (MiniLM-L6-v2)
    → Store in ChromaDB → HyDE / Multi-query retrieval
    → Generate answer (Gemini 2.0 Flash)
    → LLM-as-judge evaluation (faithfulness · relevance · completeness)
    → Embedding drift monitoring (cosine similarity over time)
    → Gradio dashboard (Q&A · Evaluation · Drift chart)
```

---

## ⚙️ Techniques covered

| Technique | Description |
|---|---|
| **Two-stage chunking** | `RecursiveCharacterTextSplitter` (1 000 chars) → `SentenceTransformersTokenTextSplitter` (256 tokens) |
| **HyDE** | Hypothetical Document Embeddings — generates a fake answer and embeds it alongside the query to improve recall |
| **Multi-query expansion** | Gemini generates 5 related sub-questions; results are deduplicated across all retrievals |
| **UMAP visualisation** | Projects 384-dim embeddings to 2-D to show how queries relate to the corpus |
| **LLM-as-judge evaluation** | Gemini scores each answer: faithfulness / relevance / completeness (0–10) |
| **Embedding drift monitoring** | Tracks cosine similarity between queries and the corpus centroid over time |
| **Gradio dashboard** | Three-tab interactive UI: Q&A · Evaluation bar chart · Live drift line chart |

---

## 🛠️ Stack

- **LLM & embeddings:** `google-genai` (Gemini 2.5 Flash)
- **Vector store:** `ChromaDB` with `all-MiniLM-L6-v2` embeddings
- **Chunking:** `langchain-text-splitters`
- **Dimensionality reduction:** `umap-learn`
- **UI:** `gradio`
- **PDF parsing:** `pypdf`

---



```python
# Cell 1 — install everything
!pip install chromadb pypdf google-genai langchain-text-splitters \
             sentence-transformers umap-learn gradio --quiet
```

---

## 📁 File structure

```
rag_annual_report_full.ipynb   ← main notebook
query_log.csv                  ← auto-generated audit log (created at runtime)
```

---

## 📊 Evaluation metrics explained

| Metric | What it catches |
|---|---|
| **Faithfulness** | Hallucinations — claims not grounded in the retrieved context |
| **Relevance** | Answers that are accurate but don't address the actual question |
| **Completeness** | Answers that are correct but leave out important available facts |

Scores are logged to `query_log.csv` on every query for offline analysis.

---

## 📡 Drift monitoring

Each query's embedding is compared to the **corpus centroid** (mean of all stored chunk vectors) using cosine similarity:

- 🟢 `> 0.30` — low drift, query is well-covered by the document
- 🟡 `0.15–0.30` — moderate drift
- 🔴 `< 0.15` — high drift, the document likely can't answer this

---

## 👤 Author

**Sanusi Isiaka Olatunji**  
Data Scientist & Electrical Engineer  
