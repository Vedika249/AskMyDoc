import streamlit as st
from rag_pipeline import RAGPipeline
from nlp_utils import extract_entities, extract_keywords, extract_topics

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AskMyDoc",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1a1a2e;
    }
    .subtitle {
        font-size: 1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .answer-box {
        background: #f0f7ff;
        border-left: 4px solid #2563eb;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .source-box {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        padding: 0.75rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    .keyword-badge {
        display: inline-block;
        background: #dbeafe;
        color: #1d4ed8;
        padding: 0.2rem 0.6rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        margin: 0.2rem;
    }
    .metric-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 0.5rem;
        padding: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ──────────────────────────────────────────────────────
if "rag" not in st.session_state:
    st.session_state.rag = None
if "doc_loaded" not in st.session_state:
    st.session_state.doc_loaded = False
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "nlp_analysis" not in st.session_state:
    st.session_state.nlp_analysis = None


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 AskMyDoc")
    st.markdown("*RAG Document Intelligence System*")
    st.divider()

    # File Upload
    st.markdown("### Upload Document")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload any PDF document to start asking questions"
    )

    if uploaded_file and not st.session_state.doc_loaded:
        with st.spinner("Loading document and building vector index..."):
            # Initialize RAG pipeline
            st.session_state.rag = RAGPipeline()
            num_chunks = st.session_state.rag.load_document(uploaded_file)
            st.session_state.doc_loaded = True
            st.success(f"✅ Document loaded! {num_chunks} chunks indexed.")

    if st.session_state.doc_loaded:
        st.divider()

        # NLP Analysis Button
        if st.button("🔍 Run NLP Analysis", use_container_width=True):
            with st.spinner("Running NLP analysis..."):
                all_text = st.session_state.rag.get_all_text()
                chunk_texts = st.session_state.rag.get_chunk_texts()

                st.session_state.nlp_analysis = {
                    "entities": extract_entities(all_text),
                    "keywords": extract_keywords(all_text, top_n=10),
                    "topics": extract_topics(chunk_texts[:20])
                }
            st.success("✅ NLP Analysis complete!")

        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        # Reset document button
        if st.button("🔄 Upload New Document", use_container_width=True):
            st.session_state.rag = None
            st.session_state.doc_loaded = False
            st.session_state.chat_history = []
            st.session_state.nlp_analysis = None
            st.rerun()

    st.divider()
    st.markdown("**Built with:**")
    st.markdown("🤗 HuggingFace · 🦜 LangChain")
    st.markdown("🗄️ FAISS · 📊 RAGAS · 🐍 spaCy")


# ─── Main Content ─────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📄 AskMyDoc</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">RAG Document Intelligence System — Ask anything about your document</p>', unsafe_allow_html=True)

# ─── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Chat", "🔍 NLP Analysis", "📊 Evaluation"])


# ── Tab 1: Chat ──────────────────────────────────────────────────────────────
with tab1:
    if not st.session_state.doc_loaded:
        st.info("👈 Please upload a PDF document from the sidebar to get started.")
    else:
        # Display chat history
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.markdown(f'<div class="answer-box">{chat["answer"]}</div>', unsafe_allow_html=True)

                # Show sources
                with st.expander("📖 View Sources"):
                    for i, source in enumerate(chat["sources"], 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Source {i} — Page {source['page']}</strong><br>
                            {source['content']}
                        </div>
                        """, unsafe_allow_html=True)

        # Chat input
        query = st.chat_input("Ask a question about your document...")
        if query:
            with st.chat_message("user"):
                st.write(query)

            with st.chat_message("assistant"):
                with st.spinner("Searching document and generating answer..."):
                    result = st.session_state.rag.get_answer(query)

                st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

                with st.expander("📖 View Sources"):
                    for i, source in enumerate(result["sources"], 1):
                        st.markdown(f"""
                        <div class="source-box">
                            <strong>Source {i} — Page {source['page']}</strong><br>
                            {source['content']}
                        </div>
                        """, unsafe_allow_html=True)

            # Save to chat history
            st.session_state.chat_history.append({
                "question": query,
                "answer": result["answer"],
                "sources": result["sources"]
            })


# ── Tab 2: NLP Analysis ──────────────────────────────────────────────────────
with tab2:
    if not st.session_state.doc_loaded:
        st.info("👈 Please upload a document first.")
    elif not st.session_state.nlp_analysis:
        st.info("👈 Click **Run NLP Analysis** in the sidebar to analyze your document.")
    else:
        analysis = st.session_state.nlp_analysis

        col1, col2 = st.columns(2)

        # Keywords
        with col1:
            st.markdown("### 🔑 Top Keywords")
            keywords_html = ""
            for kw, score in analysis["keywords"]:
                keywords_html += f'<span class="keyword-badge">{kw} ({score:.2f})</span>'
            st.markdown(keywords_html, unsafe_allow_html=True)

        # Topics
        with col2:
            st.markdown("### 📚 Main Topics")
            for i, topic in enumerate(analysis["topics"], 1):
                st.markdown(f"**{i}.** {topic}")

        st.divider()

        # Named Entities
        st.markdown("### 🏷️ Named Entities (NER)")
        entities = analysis["entities"]
        if entities:
            cols = st.columns(3)
            entity_items = list(entities.items())
            for i, (label, values) in enumerate(entity_items):
                with cols[i % 3]:
                    st.markdown(f"**{label}**")
                    for v in values[:5]:
                        st.markdown(f"- {v}")
        else:
            st.info("No named entities found in document.")


# ── Tab 3: Evaluation ────────────────────────────────────────────────────────
with tab3:
    st.markdown("### 📊 RAGAS Evaluation")
    st.markdown("Evaluate your RAG system quality using the RAGAS framework.")

    if not st.session_state.doc_loaded:
        st.info("👈 Please upload a document first.")
    else:
        st.markdown("#### Add Test Questions")
        st.markdown("Enter question-answer pairs to evaluate your RAG system:")

        with st.form("eval_form"):
            q1 = st.text_input("Question 1")
            a1 = st.text_input("Expected Answer 1")
            q2 = st.text_input("Question 2")
            a2 = st.text_input("Expected Answer 2")
            q3 = st.text_input("Question 3")
            a3 = st.text_input("Expected Answer 3")
            submitted = st.form_submit_button("▶️ Run Evaluation")

        if submitted:
            questions = [q for q in [q1, q2, q3] if q]
            ground_truths = [a for a in [a1, a2, a3] if a]

            if len(questions) < 1:
                st.warning("Please enter at least 1 question.")
            else:
                with st.spinner("Running RAGAS evaluation..."):
                    try:
                        from evaluator import evaluate_rag

                        # Generate answers and contexts for each question
                        answers = []
                        contexts = []
                        for q in questions:
                            result = st.session_state.rag.get_answer(q)
                            answers.append(result["answer"])
                            contexts.append([s["content"] for s in result["sources"]])

                        # Run RAGAS
                        scores = evaluate_rag(questions, answers, contexts, ground_truths)

                        # Display results
                        st.success("✅ Evaluation complete!")
                        col1, col2, col3, col4 = st.columns(4)

                        metrics = {
                            "Faithfulness": scores.get("faithfulness", 0),
                            "Answer Relevancy": scores.get("answer_relevancy", 0),
                            "Context Recall": scores.get("context_recall", 0),
                            "Context Precision": scores.get("context_precision", 0)
                        }

                        for col, (metric, score) in zip([col1, col2, col3, col4], metrics.items()):
                            with col:
                                st.metric(metric, f"{score:.2f}")

                    except Exception as e:
                        st.error(f"Evaluation error: {str(e)}")
                        st.info("Tip: Make sure RAGAS is installed: `pip install ragas`")
