from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class GenerateResponse:

    def __init__(self, retriever):
        """Initialize the generator with retriever and output parser"""
        self.retriever = retriever
        # raise NotImplementedError  # Commented out to allow import
        self.parser = StrOutputParser()

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
        chain =(
            {"context": self.retriever, "query": RunnablePassthrough()}
            | prompt
            | model
            | self.parser
        )
        return chain

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
