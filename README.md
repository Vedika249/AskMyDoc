# 📄 AskMyDoc — RAG Document Intelligence System

> Upload any PDF. Ask anything. Get accurate answers with source citations.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.1.20-green)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.33-red)
![LLaMA](https://img.shields.io/badge/LLaMA-3.1-purple)

---

## 🚀 Features

- **RAG Pipeline** — Retrieval-Augmented Generation using FAISS + LLaMA 3.1
- **Semantic Search** — HuggingFace sentence-transformers for dense vector embeddings
- **Cross-Encoder Re-ranking** — ms-marco-MiniLM for improved retrieval accuracy
- **NLP Analysis** — Named Entity Recognition (spaCy), Keyword Extraction (keyBERT), Topic Modeling (BERTopic)
- **Source Citations** — Every answer shows exact source page and paragraph
- **RAGAS Evaluation** — Faithfulness and Answer Relevancy metrics
- **Dark UI Dashboard** — Clean, modern Streamlit interface

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| Re-ranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| LLM | LLaMA 3.1 via Groq API |
| NLP | spaCy, keyBERT, BERTopic |
| Framework | LangChain |
| UI | Streamlit |
| Evaluation | RAGAS |

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Vedika249/AskMyDoc.git
cd AskMyDoc
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download spaCy model
```bash
python -m spacy download en_core_web_sm
```

### 5. Set Groq API Key
```bash
# Get free key from https://console.groq.com
set GROQ_API_KEY=your_key_here  # Windows
export GROQ_API_KEY=your_key_here  # Mac/Linux
```

### 6. Run the app
```bash
streamlit run app.py
```

---

## 📁 Project Structure
```
AskMyDoc/
├── app.py              # Streamlit UI (Chat, NLP Analysis, Evaluation tabs)
├── rag_pipeline.py     # Core RAG pipeline (FAISS + LLaMA 3.1)
├── nlp_utils.py        # NER, keyword extraction, topic modeling
├── evaluator.py        # RAGAS evaluation metrics
├── embeddings.py       # HuggingFace embedding setup
├── reranker.py         # Cross-encoder re-ranking
├── requirements.txt    # Dependencies
└── README.md
```

---

## 🎯 How It Works

```
PDF Upload
    ↓
Document Chunking (RecursiveCharacterTextSplitter)
    ↓
Dense Embeddings (sentence-transformers)
    ↓
FAISS Vector Store
    ↓
User Query → Semantic Search → Top-10 Chunks
    ↓
Cross-Encoder Re-ranking → Top-3 Chunks
    ↓
LLaMA 3.1 (Groq) → Answer + Source Citations
```

---

## 📊 Evaluation Metrics (RAGAS)

| Metric | Description |
|---|---|
| Faithfulness | Is the answer grounded in the document? |
| Answer Relevancy | Does the answer address the question? |

---

## 👩‍💻 Author

**Vedika Shivhare** — B.Tech CSE, Jaypee University of Engineering and Technology

[![GitHub](https://img.shields.io/badge/GitHub-Vedika249-black)](https://github.com/Vedika249)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-vedika--shivhare-blue)](https://www.linkedin.com/in/vedika-shivhare-28250233a/)