from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from typing import List
from langchain_core.documents import Document

class ArticleLoader:

    def __init__(self, path):
        """Initialise file path."""
        self.path = path

    def document_load(self) -> List[Document]:
        """
        Load articles from data/artilces.txt Text file.
        Note:
            - Must return documents.
            - Raise FileNotFoundError if the file is not found.
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"No file at {self.path}")
        docs = TextLoader(self.path, encoding="utf-8").load()
        return docs


    def create_chunks(self, documents: List[Document]) -> List[Document]:
        """
        Split the documents into chunks of size 500 and overlap of 50.
        Returns the created chunks.
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        return text_splitter.split_documents(documents)
