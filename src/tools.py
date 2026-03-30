

from langchain_core.tools import tool


def create_retriever_tool(retriever):

    @tool
    def retriever_tool(query: str) -> str:
        """Tool function that uses the retriever to fetch relevant documents based on the query."""
        docs = retriever.invoke(query)

        if not docs:
            return "No relevant information found."

        return "\n\n".join(
            [f"[Doc {i+1}]:\n{doc.page_content}" for i, doc in enumerate(docs)]
        )

    return retriever_tool
