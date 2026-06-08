import tempfile
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


class RAGPipeline:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vectorstore = None
        self.chunks = []
        self.reranker = self._load_reranker()

    def _load_reranker(self):
        try:
            from sentence_transformers import CrossEncoder
            return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception:
            return None

    def load_document(self, uploaded_file) -> int:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=512,
                chunk_overlap=50,
                separators=["\n\n", "\n", ".", " ", ""]
            )
            self.chunks = splitter.split_documents(documents)
            self.vectorstore = FAISS.from_documents(self.chunks, self.embeddings)
            return len(self.chunks)
        finally:
            os.unlink(tmp_path)

    def _rerank(self, query, chunks, top_k=3):
        if self.reranker is None:
            return chunks[:top_k]
        pairs = [(query, chunk.page_content) for chunk in chunks]
        scores = self.reranker.predict(pairs)
        scored = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored[:top_k]]

    def _generate_answer(self, prompt):
        try:
            from groq import Groq
            client = Groq()
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=256
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

    def get_answer(self, query):
        if not self.vectorstore:
            return {"answer": "Please upload a document first.", "sources": []}
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 20}
        )
        retrieved_chunks = retriever.invoke(query)
        top_chunks = self._rerank(query, retrieved_chunks, top_k=3)
        context = "\n\n".join([chunk.page_content for chunk in top_chunks])
        prompt = f"Answer based only on context below.\nContext:\n{context}\nQuestion: {query}\nAnswer:"
        answer = self._generate_answer(prompt)
        sources = []
        for chunk in top_chunks:
            sources.append({
                "page": chunk.metadata.get("page", "N/A"),
                "content": chunk.page_content[:200] + "..."
            })
        return {"answer": answer, "sources": sources, "context": context}

    def get_all_text(self):
        return " ".join([chunk.page_content for chunk in self.chunks])

    def get_chunk_texts(self):
        return [chunk.page_content for chunk in self.chunks]
