import streamlit as st
from .api_utils import upload_doc, reset_index

def render_sidebar():
    with st.sidebar:
        st.title("⚙️ Knowledge Base")
        uploaded_file = st.file_uploader("Upload PDF/DOCX", type=["pdf", "docx"])

        if st.button("Index Document"):
            if uploaded_file:
                with st.spinner("Decomposing and Indexing..."):
                    result = upload_doc(uploaded_file)
                    if result["ok"]:
                        st.success(f"{uploaded_file.name}: {result['message']}")
                    else:
                        st.error(result["message"])
            else:
                st.warning("Please upload a file first.")

        if st.button("Reset Vector Index"):
            with st.spinner("Rebuilding index from backend/docs..."):
                result = reset_index()
                if result["ok"]:
                    st.success(result["message"])
                else:
                    st.error(result["message"])
        