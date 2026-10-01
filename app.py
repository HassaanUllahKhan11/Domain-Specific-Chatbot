import os
import streamlit as st

from rag import (
    build_vector_store,
    create_rag_engine,
    get_pdf_files,
    load_vector_store,
)

st.set_page_config(
    page_title="Management Study Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Management Study Assistant")
st.caption("A local RAG chatbot that answers questions from your uploaded management PDFs.")

DEFAULT_MODEL = os.getenv("LLM_MODEL_PATH", "models/llama-2-7b-chat.Q5_K_M.gguf")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

with st.sidebar:
    st.header("Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload PDF textbook(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDFs to build the searchable knowledge base.",
    )

    index_button = st.button(
        "🔎 Build / Rebuild Index",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    )

    st.divider()
    st.header("Model")
    model_path = st.text_input("Llama GGUF path", value=DEFAULT_MODEL)
    use_gpu = st.checkbox("Use GPU acceleration", value=False)
    st.caption(
        "GPU acceleration requires a llama-cpp-python build configured for your GPU. "
        "CPU mode is the default for portability."
    )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

if index_button:
    with st.status("Building knowledge base...", expanded=True) as status:
        try:
            os.makedirs("data", exist_ok=True)
            paths = []

            for uploaded in uploaded_files:
                target = os.path.join("data", uploaded.name)
                with open(target, "wb") as f:
                    f.write(uploaded.getbuffer())
                paths.append(target)

            st.write(f"Saved {len(paths)} PDF file(s).")
            db_path = build_vector_store(paths, "vectorstore")
            st.write(f"Created FAISS index at `{db_path}`.")
            status.update(label="Knowledge base ready.", state="complete")
            st.session_state.pop("engine", None)
        except Exception as exc:
            status.update(label="Indexing failed.", state="error")
            st.exception(exc)

db_exists = os.path.exists("vectorstore/index.faiss") and os.path.exists(
    "vectorstore/index.pkl"
)

if not db_exists:
    st.info(
        "Upload a management PDF from the sidebar and click **Build / Rebuild Index** "
        "to create the searchable knowledge base."
    )
else:
    try:
        if "engine" not in st.session_state:
            with st.spinner("Loading knowledge base..."):
                vector_store = load_vector_store("vectorstore")
                st.session_state.engine = create_rag_engine(
                    vector_store=vector_store,
                    model_path=model_path,
                    use_gpu=use_gpu,
                )

        engine = st.session_state.engine

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("Sources"):
                        for source in message["sources"]:
                            st.write(source)

        question = st.chat_input("Ask a question about the uploaded textbook...")

        if question:
            st.session_state.messages.append(
                {"role": "user", "content": question}
            )
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching the textbook and generating an answer..."):
                    try:
                        answer, sources = engine.ask(question)
                        st.markdown(answer)

                        if sources:
                            with st.expander("Sources"):
                                for source in sources:
                                    st.write(source)

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": sources,
                            }
                        )
                    except Exception as exc:
                        st.error(
                            "The model could not generate an answer. "
                            "Check the GGUF model path and installation."
                        )
                        st.exception(exc)

    except FileNotFoundError as exc:
        st.error(str(exc))
    except Exception as exc:
        st.error("Could not load the RAG system.")
        st.exception(exc)
