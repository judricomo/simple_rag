from dotenv import load_dotenv
load_dotenv()

from ArticleLoader.ArticleLoader import ArticleLoader
from ArticleRetriever.ArticleRetriever import ArticleRetriever
from Response.GenerateResponse import GenerateResponse
import streamlit as st

def main():
    st.title("News Summarizer and Insight Generator")
    st.write("Ask questions about news or search for latest news")

    query = st.text_input(
        "Enter the question or search query:",
        placeholder = "e.g., 'What are some news on Chinese stocks'"
    )

    loader = ArticleLoader("data/articles.txt")
    documents = loader.document_load()
    text_chunks = loader.create_chunks(documents)
    article_retriever = ArticleRetriever()
    retriever = article_retriever.create_retriever(text_chunks)
    model = article_retriever.load_model()

    if st.button("Search") and query:
        with st.spinner("Searching for relevant news...."):
            rag = GenerateResponse(retriever)
            rag_chain = rag.create_ragchain(model)
            response = rag.generate_relevant_text(query, rag_chain)

            st.subheader("AI response")
            st.write(response)

if __name__ == "__main__":
    main()