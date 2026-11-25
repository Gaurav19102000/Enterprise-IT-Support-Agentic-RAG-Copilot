import time
from pinecone import Pinecone , ServerlessSpec
#from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from app.core.config import get_settings

settings = get_settings()


_embeddings = None
_vectorstore = None


EMBEDDING_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "all-minilm-l6-v2": 384,
}



def get_embedding_dimension() -> int:
    return 384

def get_embeddings():
    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    return _embeddings




def ensure_index():
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")

    desired_dimension = get_embedding_dimension()

    print("Embedding dimension:", desired_dimension)
    print("Configured index name:", settings.pinecone_index_name)
    print("Configured namespace:", settings.pinecone_namespace)

    pc = Pinecone(api_key=settings.pinecone_api_key)

    names = [x["name"] for x in pc.list_indexes()]
    print("Available Pinecone indexes:", names)

    if settings.pinecone_index_name in names:
        index_info = pc.describe_index(settings.pinecone_index_name)
        current_dimension = getattr(index_info, "dimension", None)

        if current_dimension is None and isinstance(index_info, dict):
            current_dimension = index_info.get("dimension")

        print("Existing index dimension:", current_dimension)

        if current_dimension != desired_dimension:
            raise RuntimeError(
                f"Index dimension mismatch! Existing: "
                f"{current_dimension}, Required: {desired_dimension}. "
                "Create a new 384-dimensional index or verify your "
                "configured index name."
            )

    else:
        print("Configured index does not exist. Creating new index...")

        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=desired_dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
        )

        while not pc.describe_index(
            settings.pinecone_index_name
        ).status["ready"]:
            time.sleep(1)

    return pc.Index(settings.pinecone_index_name)

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        index = ensure_index()
        _vectorstore = PineconeVectorStore(
            index=index,
            embedding=get_embeddings(),
            namespace=settings.pinecone_namespace,
        )
    return _vectorstore


def get_retriever():
    return get_vectorstore().as_retriever(search_kwargs={"k": settings.top_k})



def add_documents(chunks):
    store = get_vectorstore()
    return store.add_documents(chunks)