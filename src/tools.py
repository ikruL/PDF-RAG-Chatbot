

from langchain_core.tools import tool


def create_retriever_tool(retriever):

    @tool
    def retriever_tool(query: str):
        """Fetch relevant documents"""

        docs = retriever.invoke(query)

        if not docs:
            return {
                "content": "No relevant information found.",
                "contexts": []
            }

        contexts = [doc.page_content for doc in docs]

        return {
            "content": "\n\n".join(
                [f"[Doc {i+1}]:\n{c}" for i, c in enumerate(contexts)]
            ),
            "contexts": contexts
        }

    return retriever_tool
