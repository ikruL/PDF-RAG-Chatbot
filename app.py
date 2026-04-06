
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from src.loader import load_pdfs
from src.tools import create_retriever_tool
from src.retriever import build_vectorstore, create_retriever
from src.embeddings import get_embeddings
from src.agent_graph import create_agent_graph

load_dotenv()  # Load environment variables from .env file

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "ready" not in st.session_state:
    st.session_state.ready = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever_tool" not in st.session_state:
    st.session_state.retriever_tool = None

st.title("RAG Agent")


with st.sidebar:

    max_tokens = st.slider("Max tokens", 0, 1000, value=700)

    clear = st.button("Clear Conversation", type="secondary",
                      use_container_width=True)
    if clear:
        st.session_state.messages = []
    st.divider()

    gemini_api_key = st.text_input("Gemini API Key", type="password")

    if gemini_api_key.strip():
        st.session_state.gemini_api_key = gemini_api_key.strip()

uploaded_files = st.file_uploader(
    "Upload PDF Files", type="pdf", accept_multiple_files=True)

if uploaded_files:
    current_files = [f.name for f in uploaded_files]

    if "last_uploaded" not in st.session_state:
        st.session_state.last_uploaded = current_files

    elif st.session_state.last_uploaded != current_files:
        st.session_state.vectorstore = None
        st.session_state.retriever_tool = None
        st.session_state.ready = False
        st.session_state.messages = []

        st.session_state.last_uploaded = current_files


if st.button("Load PDF Files", type="primary"):
    api_key = st.session_state.get("gemini_api_key")
    if not api_key:
        st.warning("Please enter API key")
        st.stop()

    if not uploaded_files:
        st.warning("Please upload at least one PDF")
        st.stop()

    with st.spinner("Processing..."):
        docs = load_pdfs(uploaded_files)
        embeddings = get_embeddings(gemini_api_key)
        vectorstore = build_vectorstore(
            docs, embeddings)
        retriever = create_retriever(vectorstore)
        retriever_tool = create_retriever_tool(retriever)

        st.session_state.vectorstore = vectorstore
        st.session_state.retriever_tool = retriever_tool
        st.session_state.ready = True

    st.success("Documents processed!")
    print("Docs sample:", docs[0].page_content[:200])


def format_response(response):

    if isinstance(response, list):
        text = "\n".join([r.get("text", "") for r in response])
    else:
        text = str(response)

    text = text.replace("\n", "\n\n")

    return text


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if "ready" not in st.session_state:
    st.session_state.ready = False

if not st.session_state.get("gemini_api_key"):
    st.info("Enter API key in sidebar")

elif not st.session_state.get("ready"):
    st.info("Upload and process documents first")
else:
    user_input = st.chat_input(
        "Ask something about your PDFs...",
        disabled=not st.session_state.get("ready", False)
    )
    if user_input and st.session_state.ready:
        api_key = st.session_state.get("gemini_api_key")
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        try:
            messages = [HumanMessage(content=user_input)]
            retriever_tool = st.session_state.get("retriever_tool")
            graph = create_agent_graph(
                api_key, max_tokens=max_tokens, tools=[retriever_tool])
            with st.spinner("Thinking..."):
                response = graph.invoke(
                    {"messages": messages})

            answer = format_response(response["messages"][-1].content)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            with st.chat_message("assistant"):
                st.markdown(answer)

        except Exception as e:
            st.error(f" Error: {e}")
