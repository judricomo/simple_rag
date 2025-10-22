# Simple RAG (Retrieval-Augmented Generation) News Summarizer

A Streamlit-based application that uses RAG (Retrieval-Augmented Generation) to answer questions about news articles. This project demonstrates how to build an intelligent question-answering system using LangChain, OpenAI, and FAISS vector store.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [How to Run Locally](#how-to-run-locally)
- [Code Explanation](#code-explanation)
  - [main.py](#mainpy)
  - [ArticleLoader.py](#articleloaderpy)
  - [ArticleRetriever.py](#articleretrieverpy)
  - [GenerateResponse.py](#generateresponsepy)
- [How It Works](#how-it-works)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

## 🔍 Overview

This application implements a RAG (Retrieval-Augmented Generation) system that:
1. Loads news articles from a text file
2. Splits them into manageable chunks
3. Creates embeddings and stores them in a FAISS vector database
4. Retrieves relevant context based on user queries
5. Uses OpenAI's GPT-4o-mini to generate intelligent responses

## ✨ Features

- **Document Loading**: Automatically loads and processes text documents
- **Text Chunking**: Intelligently splits documents into chunks with overlap
- **Vector Search**: Uses FAISS for efficient similarity search
- **OpenAI Integration**: Leverages GPT-4o-mini for response generation
- **Streamlit UI**: Clean and interactive web interface
- **Context-Aware Responses**: Provides answers based on retrieved relevant articles

## 📁 Project Structure

```
simple_rag/
├── main.py                          # Main Streamlit application
├── ArticleLoader/
│   ├── __init__.py
│   └── ArticleLoader.py            # Document loading and chunking
├── ArticleRetriever/
│   ├── __init__.py
│   └── ArticleRetriever.py         # Vector store and retriever
├── Response/
│   ├── __init__.py
│   └── GenerateResponse.py         # RAG chain and response generation
├── data/
│   └── articles.txt                # Source news articles
└── README.md                        # This file
```

## 🔧 Prerequisites

- Python 3.8 or higher
- OpenAI API key
- pip (Python package installer)

## 📦 Installation

1. **Clone or download the repository:**
   ```bash
   cd /path/to/simple_rag
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install required packages:**
   ```bash
   pip install streamlit langchain langchain-community langchain-openai langchain-text-splitters faiss-cpu python-dotenv
   ```

   **Package descriptions:**
   - `streamlit`: Web application framework for the UI
   - `langchain`: Framework for building LLM applications
   - `langchain-community`: Community integrations for LangChain
   - `langchain-openai`: OpenAI integration for LangChain
   - `langchain-text-splitters`: Text splitting utilities
   - `faiss-cpu`: Facebook AI Similarity Search for vector storage
   - `python-dotenv`: Environment variable management

## ⚙️ Configuration

1. **Create a `.env` file** in the project root directory:
   ```bash
   touch .env
   ```

2. **Add your OpenAI API key** to the `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```

   To get an OpenAI API key:
   - Go to [OpenAI Platform](https://platform.openai.com/)
   - Sign up or log in
   - Navigate to API keys section
   - Create a new secret key

3. **Ensure articles data exists:**
   - Make sure `data/articles.txt` contains your news articles
   - Each article should be separated by a blank line

## 🚀 How to Run Locally

1. **Navigate to the project directory:**
   ```bash
   cd /Users/juandavid.ricomolano/Library/CloudStorage/OneDrive-Slalom/Documents/simple_rag
   ```

2. **Activate your virtual environment** (if using one):
   ```bash
   source venv/bin/activate  # On macOS/Linux
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run main.py
   ```

4. **Access the application:**
   - The application will automatically open in your default browser
   - If not, navigate to: `http://localhost:8501`

5. **Use the application:**
   - Enter a question in the text input (minimum 10 characters)
   - Click "Search" button
   - View the AI-generated response based on relevant articles

## 📖 Code Explanation

### main.py

The entry point of the application that sets up the Streamlit interface and orchestrates the RAG pipeline.

```python
from dotenv import load_dotenv
load_dotenv()
```
**Lines 1-2:** Import and load environment variables from `.env` file (including OpenAI API key).

```python
from ArticleLoader.ArticleLoader import ArticleLoader
from ArticleRetriever.ArticleRetriever import ArticleRetriever
from Response.GenerateResponse import GenerateResponse
import streamlit as st
```
**Lines 4-7:** Import custom modules and Streamlit framework.
- `ArticleLoader`: Handles document loading and text splitting
- `ArticleRetriever`: Creates vector store and retriever
- `GenerateResponse`: Builds RAG chain and generates responses
- `streamlit`: Web framework for the user interface

```python
def main():
    st.title("News Summarizer and Insight Generator")
    st.write("Ask questions about news or search for latest news")
```
**Lines 9-11:** Define main function and create Streamlit UI title and description.

```python
    query = st.text_input(
        "Enter the question or search query:",
        placeholder = "e.g., 'What are some news on Chinese stocks'"
    )
```
**Lines 13-16:** Create a text input field where users can enter their questions. Includes a placeholder example.

```python
    loader = ArticleLoader("data/articles.txt")
    documents = loader.document_load()
    text_chunks = loader.create_chunks(documents)
```
**Lines 18-20:**
- **Line 18:** Initialize `ArticleLoader` with the path to articles file
- **Line 19:** Load all documents from the text file
- **Line 20:** Split documents into smaller chunks (500 chars each, 50 char overlap)

```python
    article_retriever = ArticleRetriever()
    retriever = article_retriever.create_retriever(text_chunks)
    model = article_retriever.load_model()
```
**Lines 21-23:**
- **Line 21:** Initialize `ArticleRetriever` (sets up OpenAI embeddings)
- **Line 22:** Create FAISS vector store from chunks and return retriever (retrieves top 4 similar docs)
- **Line 23:** Load GPT-4o-mini model with temperature=0 (deterministic responses)

```python
    if st.button("Search") and query:
        with st.spinner("Searching for relevant news...."):
```
**Lines 25-26:** 
- **Line 25:** Check if "Search" button is clicked and query is not empty
- **Line 26:** Display a spinner with message while processing

```python
            rag = GenerateResponse(retriever)
            rag_chain = rag.create_ragchain(model)
            response = rag.generate_relevant_text(query, rag_chain)
```
**Lines 27-29:**
- **Line 27:** Initialize response generator with retriever
- **Line 28:** Create the RAG chain (retriever → prompt → model → parser)
- **Line 29:** Generate response by invoking the chain with user query

```python
            st.subheader("AI response")
            st.write(response)
```
**Lines 31-32:** Display the AI-generated response in the Streamlit interface.

```python
if __name__ == "__main__":
    main()
```
**Lines 34-35:** Execute main function when script is run directly.

---

### ArticleLoader/ArticleLoader.py

Handles loading documents from file and splitting them into chunks for efficient processing.

```python
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from typing import List
from langchain_core.documents import Document
```
**Lines 1-5:** Import necessary dependencies:
- `TextLoader`: Loads text from files
- `RecursiveCharacterTextSplitter`: Splits text intelligently
- `os`: File system operations
- `List`, `Document`: Type hints for better code clarity

```python
class ArticleLoader:

    def __init__(self, path):
        """Initialise file path."""
        self.path = path
```
**Lines 7-11:** Define `ArticleLoader` class with constructor.
- **Line 10:** Store the file path as an instance variable

```python
    def document_load(self) -> List[Document]:
        """
        Load articles from data/artilces.txt Text file.
        Note:
            - Must return documents.
            - Raise FileNotFoundError if the file is not found.
        """
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"No file at {self.path}")
```
**Lines 13-21:** Define method to load documents.
- **Line 13:** Return type is a list of Document objects
- **Lines 19-20:** Check if file exists; raise error if not found

```python
        docs = TextLoader(self.path, encoding="utf-8").load()
        return docs
```
**Lines 22-23:**
- **Line 22:** Use LangChain's TextLoader to read file with UTF-8 encoding
- **Line 23:** Return the loaded documents

```python
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
```
**Lines 26-35:** Define method to split documents into chunks.
- **Line 26:** Accept list of documents, return list of chunked documents
- **Lines 30-33:** Create text splitter with:
  - `chunk_size=500`: Each chunk contains max 500 characters
  - `chunk_overlap=50`: 50 characters overlap between consecutive chunks (maintains context)
- **Line 34:** Split all documents and return the chunks

**Why chunking?** Large documents are split into smaller pieces to:
1. Fit within embedding model limits
2. Enable more precise retrieval of relevant information
3. Improve search accuracy by matching specific sections

---

### ArticleRetriever/ArticleRetriever.py

Creates embeddings, builds a vector store, and provides retrieval capabilities.

```python
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.vectorstores.base import VectorStoreRetriever
```
**Lines 1-3:** Import necessary components:
- `OpenAIEmbeddings`: Creates vector embeddings using OpenAI's models
- `ChatOpenAI`: OpenAI's chat model interface
- `FAISS`: Facebook AI Similarity Search - efficient vector storage
- `VectorStoreRetriever`: Interface for retrieving documents

```python
class ArticleRetriever:

    def __init__(self):
        """Initialize vector store with OpenAI embeddings."""
        self.embeddings = OpenAIEmbeddings()
```
**Lines 5-9:** Define class and constructor.
- **Line 9:** Initialize OpenAI embeddings model (uses API key from environment)
- This model converts text into numerical vectors for similarity comparison

```python
    def create_retriever(self, text_chunks) -> VectorStoreRetriever:
        """Create FAISS vector store from documents. Use the vector store to create and return a retriever that returns 4 most relevant documents"""
        vs = FAISS.from_documents(text_chunks, self.embeddings)
        return vs.as_retriever(search_kwargs={"k": 4})
```
**Lines 11-14:** Create retriever from document chunks.
- **Line 13:** Create FAISS vector store by:
  1. Converting all text chunks to embeddings
  2. Storing them in an efficient searchable index
- **Line 14:** Convert vector store to retriever that returns top 4 most similar documents
  - `k=4`: Number of documents to retrieve per query

**How it works:** When you search, the retriever:
1. Converts your query to an embedding
2. Finds the 4 most similar chunk embeddings
3. Returns the corresponding text chunks

```python
    def load_model(self) -> ChatOpenAI:
        """Initialize and return llm model with gpt-4o-mini"""
        return ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
```
**Lines 16-18:** Initialize the language model.
- **Line 18:** Create ChatOpenAI instance with:
  - `model_name="gpt-4o-mini"`: Uses GPT-4o-mini model (fast and cost-effective)
  - `temperature=0`: Deterministic output (same input → same output)
  - Higher temperature = more creative/random responses

---

### Response/GenerateResponse.py

Builds the RAG chain and generates responses by combining retrieval with generation.

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
```
**Lines 1-3:** Import LangChain components:
- `ChatPromptTemplate`: Creates structured prompts for the LLM
- `StrOutputParser`: Converts model output to string format
- `RunnablePassthrough`: Passes data through the chain unchanged

```python
class GenerateResponse:

    def __init__(self, retriever):
        """Initialize the generator with retriever and output parser"""
        self.retriever = retriever
        # raise NotImplementedError  # Commented out to allow import
        self.parser = StrOutputParser()
```
**Lines 5-11:** Define class and constructor.
- **Line 9:** Store the retriever for fetching relevant documents
- **Line 11:** Initialize string output parser to convert model responses to text

```python
    def create_ragchain(self, model):
        """
        This function should return a chain object with the following structure:

        Returns:
            A RAG chain object with the structure described above. The chain should be configurable to process a question by:
              1. Using the retriever to fetch relevant context
              2. Combining the context with the query
              3. Passing this to the language model

        Note: Ensure that the first part of the chain includes both 'context' and 'query' keys.
        """
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant that provides concise answers based on the provided context.\n\n"
            "Context: {context}\n\n"
            "Question: {query}\n\n"
            "Answer:"
        )
```
**Lines 13-24:** Create the RAG chain.
- **Lines 18-23:** Define the prompt template with placeholders:
  - `{context}`: Will be filled with retrieved documents
  - `{query}`: Will be filled with user's question
  - Template instructs the model to answer based on provided context

```python
        chain =(
            {"context": self.retriever, "query": RunnablePassthrough()}
            | prompt
            | model
            | self.parser
        )
        return chain
```
**Lines 25-31:** Build and return the RAG chain using pipe operator (`|`).

**Line 26:** Create dictionary with:
- `"context"`: Retriever fetches relevant documents based on query
- `"query"`: RunnablePassthrough() passes the original query unchanged

**Line 27:** Pipe to prompt template (fills {context} and {query})

**Line 28:** Pipe to language model (generates response)

**Line 29:** Pipe to parser (converts model output to string)

**Chain flow:**
```
User Query → Retriever (gets context) → Prompt Template → LLM → Parser → Final Answer
```

```python
    def generate_relevant_text(self, query: str, rag_chain) -> str:
        """
        Query the RAG chain to get response and return it

        IMPORTANT:
        - Empty queries or queries shorter than 10 characters MUST be rejected with ValueError
        - The exact error message should be: "Query too short."

        Raises:
            ValueError: If query is empty or too short (less than 10 characters)
        """
        if not query or len(query) < 10:
            raise ValueError("Query too short.")
        return rag_chain.invoke(query)
```
**Lines 33-46:** Generate response with validation.
- **Line 44:** Validate query is not empty and has at least 10 characters
- **Line 45:** Raise error if validation fails
- **Line 46:** Invoke the RAG chain with the query and return the response

**Why 10 characters?** Ensures meaningful queries that can be properly processed.

---

## 🔄 How It Works

### The RAG Pipeline

1. **Document Loading** (`ArticleLoader`)
   - Reads articles from `data/articles.txt`
   - Splits into 500-character chunks with 50-character overlap

2. **Embedding & Indexing** (`ArticleRetriever`)
   - Converts text chunks to vector embeddings using OpenAI
   - Stores embeddings in FAISS vector database

3. **Query Processing** (`GenerateResponse`)
   - User enters question
   - Query is converted to embedding
   - FAISS finds 4 most similar article chunks
   - Relevant context is combined with query in prompt
   - GPT-4o-mini generates contextual answer
   - Response is displayed to user

### Data Flow Diagram

```
┌──────────────┐
│ articles.txt │
└──────┬───────┘
       │ Load & Split
       ▼
┌──────────────────┐
│ Text Chunks      │
│ (500 chars each) │
└──────┬───────────┘
       │ Create Embeddings
       ▼
┌──────────────────┐
│ FAISS Vector DB  │
└──────┬───────────┘
       │
       │ ┌──────────────┐
       │ │ User Query   │
       │ └──────┬───────┘
       │        │
       ▼        ▼
┌──────────────────────┐
│ Retrieve Top 4       │
│ Similar Chunks       │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Combine with Prompt  │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ GPT-4o-mini          │
│ Generate Response    │
└──────┬───────────────┘
       │
       ▼
┌──────────────────────┐
│ Display to User      │
└──────────────────────┘
```

## 💡 Usage Examples

### Example 1: Asking about specific news

**Query:** "What are some news about Madagascar?"

**Expected behavior:**
- System retrieves chunks mentioning Madagascar
- GPT-4o-mini generates summary based on retrieved context
- Response might include: "President Michael Randrianirina appointed Herintsalama Rajaonarivelo as prime minister..."

### Example 2: Asking about regional topics

**Query:** "Tell me about the Mekong River patrol"

**Expected behavior:**
- Retrieves relevant context about Mekong River
- Generates response about joint patrol by China, Laos, Myanmar, and Thailand

### Example 3: Energy-related queries

**Query:** "What is happening with Russian energy in Europe?"

**Expected behavior:**
- Retrieves articles about EU energy policies and Russian gas
- Provides comprehensive answer about REPowerEU and Hungary's position

## 🐛 Troubleshooting

### Common Issues

**Issue:** "Query too short" error
- **Solution:** Ensure your query is at least 10 characters long

**Issue:** OpenAI API error
- **Solution:** 
  - Check if `.env` file exists with valid `OPENAI_API_KEY`
  - Verify API key is active and has credits
  - Check internet connection

**Issue:** Module not found error
- **Solution:** Install missing packages:
  ```bash
  pip install streamlit langchain langchain-community langchain-openai langchain-text-splitters faiss-cpu python-dotenv
  ```

**Issue:** FileNotFoundError for articles.txt
- **Solution:** Ensure `data/articles.txt` exists and contains text

**Issue:** FAISS installation fails on M1/M2 Mac
- **Solution:** Use conda instead:
  ```bash
  conda install -c conda-forge faiss-cpu
  ```

**Issue:** Streamlit doesn't open automatically
- **Solution:** Manually open browser and go to `http://localhost:8501`

## 📝 Notes

- **API Costs:** This application uses OpenAI's API which incurs costs. Monitor your usage on the OpenAI dashboard.
- **Performance:** First query may take longer as embeddings are created. Subsequent queries are faster.
- **Customization:** You can modify chunk size, overlap, number of retrieved documents (k), and model parameters in the respective classes.
- **Data Updates:** To add more articles, simply append them to `data/articles.txt` and restart the application.

## 🔐 Security

- Never commit your `.env` file to version control
- Keep your OpenAI API key secure
- Add `.env` to `.gitignore`:
  ```bash
  echo ".env" >> .gitignore
  ```

## 📚 Additional Resources

- [LangChain Documentation](https://python.langchain.com/)
- [OpenAI API Documentation](https://platform.openai.com/docs)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)

---

**Happy coding! 🚀**
