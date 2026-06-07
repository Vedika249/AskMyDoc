from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embeddings():
    """
    Load HuggingFace sentence-transformers embedding model.
    Model: all-MiniLM-L6-v2 (lightweight, fast, good quality)
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    return embeddings
