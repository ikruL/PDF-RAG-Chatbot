from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import config
import tempfile


def load_pdfs(uploaded_files):
    """Loads PDFs to ChromaDB """
    all_docs = []

    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.getbuffer())
            tmp_path = tmp.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        all_docs.extend(docs)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP)
    splits = text_splitter.split_documents(all_docs)
    return splits
