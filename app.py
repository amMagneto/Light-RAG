import streamlit as st
from frontend.components.sidebar import render_sidebar
from frontend.components.chat_interface import render_chat_interface

st.set_page_config(
    page_title="ChainReaction",
    page_icon="⚛️",
    layout="wide"
)

# Call modular components
render_sidebar()
render_chat_interface()