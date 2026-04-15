import streamlit as st
from .api_utils import upload_doc

def render_sidebar():
    with st.sidebar:
        st.title("⚙️ Knowledge Base")
        uploaded_file = st.file_uploader("Upload PDF/DOCX", type=["pdf", "docx"])

        if st.button("Index Document"):
            if uploaded_file:
                with st.spinner("Decomposing and Indexing..."):
                    success = upload_doc(uploaded_file)
                    if success:
                        st.success(f"Indexed {uploaded_file.name}!")
                    else:
                        st.error("Index failed.")
            else:
                st.warning("Please upload a file first.")
        