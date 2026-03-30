from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
from src.config import config


def get_embeddings(api_key):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            google_api_key=api_key
        )

        return embeddings

    except Exception as e:

        if "API key" in str(e) or "401" in str(e):
            st.error(
                "Invalid Gemini API Key. Please check your key and try again.")
        else:
            st.error("Failed to initialize embeddings")

        st.stop()
