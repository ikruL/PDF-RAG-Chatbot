import streamlit as st
from src.config import config
from langchain_chroma import Chroma


@st.cache_resource
def build_vectorstore(_docs, _embeddings):
    """Builds the vectorstore from the documents and embeddings"""
    try:
        vectorstore = Chroma.from_documents(
            documents=_docs, embedding=_embeddings)
        return vectorstore
    except Exception as e:
        print(f"Error setting up ChromaDB : str{e}")
        raise


def create_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.RETRIEVER_K}
    )
