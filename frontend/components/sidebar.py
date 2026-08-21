import streamlit as st
from .api_utils import upload_doc, reindex, list_documents, delete_document, health

SUPPORTED_TYPES = ["pdf", "docx", "txt", "md"]


def render_sidebar():
    with st.sidebar:
        st.title("⚙️ Knowledge Base")
        uploaded_file = st.file_uploader("Upload PDF / DOCX / TXT / MD", type=SUPPORTED_TYPES)

        if st.button("Index Document"):
            if uploaded_file:
                with st.spinner("Parsing → chunking → embedding → indexing..."):
                    result = upload_doc(uploaded_file)
                    if result["ok"]:
                        st.success(f"{uploaded_file.name}: {result['message']}")
                    else:
                        st.error(result["message"])
            else:
                st.warning("Please upload a file first.")

        status = health()
        if status:
            llm = "online" if status.get("llm_available") else "offline"
            st.caption(
                f"Backend: `{status['status']}` | LLM: **{llm}** | "
                f"Docs: **{status.get('documents_ready', 0)}** | Chunks: **{status.get('indexed_chunks', 0)}**"
            )
        else:
            st.caption("Backend unreachable — start it with `uvicorn app.main:app --reload`")

        st.divider()
        st.subheader("Documents")
        for doc in list_documents():
            meta = doc["meta"]
            col1, col2 = st.columns([4, 1])
            col1.markdown(f"**{meta['filename']}**  \n`{meta['status']}` · {doc['chunk_count']} chunks")
            if col2.button("🗑", key=f"del_{meta['id']}"):
                delete_document(meta["id"])
                st.rerun()

        if st.button("Rebuild Vector Index"):
            with st.spinner("Re-embedding all stored chunks..."):
                result = reindex()
                if result["ok"]:
                    st.success(result["message"])
                else:
                    st.error(result["message"])
