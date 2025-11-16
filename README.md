Explainable Dual-Stage Retrieval-Augmented Generation (RAG)

This project implements an explainable dual-stage RAG workflow for QA on noisy or low-resource text. It designed for medical question-answering, combining semantic and keyword-based search with cross-encoder reranking and FLAN-T5 generation.

This project implements an end-to-end RAG pipeline that retrieves relevant medical information from a healthcare conversation dataset and generates accurate responses to patient queries. The system uses state-of-the-art NLP techniques including hybrid search, Reciprocal Rank Fusion (RRF), and cross-encoder reranking for optimal retrieval performance.

# Medical RAG System with Hybrid Search

A sophisticated Retrieval-Augmented Generation (RAG) system designed for medical question-answering, combining semantic and keyword-based search with cross-encoder reranking and FLAN-T5 generation.

## 🎯 Project Overview

This project implements an end-to-end RAG pipeline that retrieves relevant medical information from a healthcare conversation dataset and generates accurate responses to patient queries. The system uses state-of-the-art NLP techniques including hybrid search, Reciprocal Rank Fusion (RRF), and cross-encoder reranking for optimal retrieval performance.

## 🏗️ System Architecture
```
User Query
    ↓
┌─────────────────────────────────────────┐
│         Hybrid Search Layer             │
│  ┌──────────────┐  ┌──────────────┐    │
│  │   Semantic   │  │   Keyword    │    │
│  │ Search (FAISS│  │ Search (BM25)│    │
│  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Reciprocal Rank Fusion (RRF)          │
│  Combines rankings from both searches   │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Cross-Encoder Reranking                │
│  Final precision ranking                 │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Answer Generation (FLAN-T5)            │
│  Generates/Extracts final answer        │
└─────────────────────────────────────────┘
    ↓
Final Answer
```

## 📊 Dataset

**HealthCareMagic-100k-en.jsonl**: A medical conversation dataset containing patient questions and doctor responses.

- Format: JSONL with conversation text
- Structure: `<human>: [patient question] <bot>: [doctor answer]`

## 🔧 Key Components

### 1. **Data Processing & Chunking**
- Parses medical conversations from JSONL format
- Splits conversations into Q&A pairs
- Each chunk contains:
  - Patient question
  - Doctor's answer
  - Metadata (chunk_id, conversation_id)

### 2. **Embedding Generation**
- **Model**: `sentence-transformers/all-MiniLM-L6-v2`
- Generates 384-dimensional embeddings
- Combines question + answer for semantic representation
- Normalized embeddings for cosine similarity

### 3. **Vector Storage (FAISS)**
- **Index Type**: IndexFlatIP (Inner Product for cosine similarity)
- Stores embeddings
- Fast similarity search in sub-second response time

### 4. **Keyword Search (BM25)**
- **Algorithm**: BM25Okapi
- Token-based retrieval for exact keyword matching
- Complements semantic search with lexical matching

### 5. **Hybrid Search with RRF**
- Combines semantic and keyword search results
- **Reciprocal Rank Fusion (RRF) Formula**:
```
  score(d) = Σ [1 / (k + rank(d))]
```
  where k=60 (tunable constant)
- Removes duplicates and creates unified ranking

### 6. **Cross-Encoder Reranking**
- **Model**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Reranks top candidates from RRF
- Provides final precision ranking
- Considers query-document interaction

### 7. **Answer Generation**
- **Model**: `google/flan-t5-base` (250M parameters)
- Generates answers based on retrieved context
- Fallback to extractive approach if generation fails
- Quality checks for answer validation

## 🚀 Pipeline Flow

1. **Query Input**: User submits medical question
2. **Semantic Search**: FAISS retrieves top-20 similar chunks
3. **Keyword Search**: BM25 retrieves top-20 keyword-matched chunks
4. **RRF Fusion**: Combines both rankings using reciprocal rank fusion
5. **Reranking**: Cross-encoder reranks top-30 candidates
6. **Context Selection**: Top-1 chunk selected as context
7. **Answer Generation**: FLAN-T5 generates answer from context
8. **Output**: Final answer with metadata


## 🛠️ Technologies Used

| Component | Technology |
|-----------|-----------|
| Embeddings | SentenceTransformers |
| Vector DB | FAISS |
| Keyword Search | BM25Okapi |
| Reranking | CrossEncoder |
| Generation | FLAN-T5 |
| Framework | PyTorch |
| Language | Python |

## 📦 Installation
```bash
# Install required packages
pip install sentence-transformers faiss-cpu rank-bm25 transformers torch

# For GPU support
pip install sentence-transformers faiss-gpu transformers torch
```

## 🎯 Key Features

✅ **Hybrid Search**: Combines semantic and lexical retrieval  
✅ **RRF Fusion**: Advanced ranking fusion algorithm  
✅ **Cross-Encoder Reranking**: Precision ranking layer  
✅ **Medical Domain**: Specialized for healthcare Q&A  
✅ **Fallback Mechanisms**: Extractive fallback if generation fails  
✅ **Quality Validation**: Automatic answer quality checks  


## 📚 References

- [SentenceTransformers](https://www.sbert.net/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [BM25](https://en.wikipedia.org/wiki/Okapi_BM25)
- [FLAN-T5](https://huggingface.co/google/flan-t5-base)
- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)



🔗 Download [dataset](https://huggingface.co/datasets/RafaelMPereira/HealthCareMagic-100k-Chat-Format-en)



