# 📄 AskMyDoc — RAG Document Intelligence System

An end-to-end Retrieval-Augmented Generation (RAG) system that lets you upload any PDF and ask questions about it using state-of-the-art NLP and ML techniques.

## 🚀 Features
- **RAG Pipeline** — HuggingFace embeddings + FAISS vector search + Flan-T5 generation
- **Cross-Encoder Re-ranking** — Improves retrieval accuracy using ms-marco-MiniLM
- **NLP Analysis** — Named Entity Recognition (spaCy), Keyword Extraction (keyBERT), Topic Modeling (BERTopic)
- **Source Citations** — Every answer shows the source page and paragraph
- **RAGAS Evaluation** — Faithfulness, Answer Relevancy, Context Recall metrics

## 🛠️ Tech Stack
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`
- **Vector Store:** FAISS
- **Re-ranker:** `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **LLM:** `google/flan-t5-large`
- **NLP:** spaCy, keyBERT, BERTopic
- **Framework:** LangChain
- **UI:** Streamlit
- **Evaluation:** RAGAS

## ⚙️ Setup & Run

```bash
# 1. Clone the repo
git clone https://github.com/Vedika249/AskMyDoc.git
cd AskMyDoc

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download spaCy model
python -m spacy download en_core_web_sm

# 4. Run the app
streamlit run app.py
```

## 📁 Project Structure
```
AskMyDoc/
├── app.py              # Streamlit UI
├── rag_pipeline.py     # Core RAG pipeline
├── embeddings.py       # HuggingFace embeddings
├── reranker.py         # Cross-encoder re-ranking
├── nlp_utils.py        # NER, keywords, topics
├── evaluator.py        # RAGAS evaluation
└── requirements.txt    # Dependencies
```

## 👩‍💻 Author
Vedika Shivhare — [GitHub](https://github.com/Vedika249)
