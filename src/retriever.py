import streamlit as st
from src.config import config
from langchain_chroma import Chroma


def build_vectorstore(docs, embeddings):
    """Builds the vectorstore from the documents and embeddings"""
    try:
        vectorstore = Chroma.from_documents(
            documents=docs, embedding=embeddings)
        return vectorstore
    except Exception as e:
        print(f"Error setting up ChromaDB : str{e}")
        raise


def create_retriever(vectorstore):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.RETRIEVER_K}
    )
