from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores.base import VectorStoreRetriever

class ArticleRetriever:

    def __init__(self):
        """Initialize vector store with OpenAI embeddings."""
        self.embeddings = OpenAIEmbeddings()

    def create_retriever(self, text_chunks) -> VectorStoreRetriever:
        """Create FAISS vector store from documents. Use the vector store to create and return a retriever that returns 4 most relevant documents"""
        vs = FAISS.from_documents(text_chunks, self.embeddings)
        return vs.as_retriever(search_kwargs={"k": 4})

    def load_model(self) -> ChatOpenAI:
        """Initialize and return llm model with gpt-4o-mini"""
        return ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
