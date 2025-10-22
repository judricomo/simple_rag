from dotenv import load_dotenv
load_dotenv()

from ArticleLoader.ArticleLoader import ArticleLoader
from ArticleRetriever.ArticleRetriever import ArticleRetriever
from Response.GenerateResponse import GenerateResponse
import streamlit as st

def main():
    st.title("🌍 Global News Intelligence Hub")
    st.write("Powered by AI • Discover insights from world events in seconds")

    query = st.text_input(
        "💬 What would you like to know about current events?",
        placeholder = "e.g., 'What's happening with energy policies in Europe?' or 'Tell me about developments in Asia'"
    )

    loader = ArticleLoader("data/articles.txt")
    documents = loader.document_load()
    text_chunks = loader.create_chunks(documents)
    article_retriever = ArticleRetriever()
    retriever = article_retriever.create_retriever(text_chunks)
    model = article_retriever.load_model()

    if st.button("🔍 Analyze News", type="primary") and query:
        with st.spinner("🤖 AI is analyzing global news sources..."):
            rag = GenerateResponse(retriever)
            rag_chain = rag.create_ragchain(model)
            response = rag.generate_relevant_text(query, rag_chain)

            st.subheader("📊 Intelligence Report")
            st.write(response)

if __name__ == "__main__":
    main()