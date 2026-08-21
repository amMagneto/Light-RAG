import streamlit as st

def inject_custom_css():
    st.markdown("""
    <style>
    /* Global Overrides */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Space Mono', monospace !important;
        color: #00FFC2 !important; /* Neon Cyan */
    }

    .main {
        background-color: #0F0F0F; /* Brutalist Black */
        background-image: radial-gradient(#FF00F5 0.5px, transparent 0.5px); /* Pixie Dust */
        background-size: 30px 30px;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFB7FF !important; /* Pastel Pixie Pink */
        border-right: 5px solid #000000 !important;
    }

    [data-testid="stSidebar"] * {
        color: #000000 !important;
        font-weight: 700 !important;
    }

    /* Titles and Headers */
    h1, h2, h3 {
        color: #FF00F5 !important; /* Hot Pink */
        text-transform: uppercase;
        text-shadow: 3px 3px 0px #00FFC2; /* Brutalist Shadow */
        border: 3px solid #FF00F5;
        padding: 10px;
        display: inline-block;
        background: #000;
    }

    /* Buttons */
    .stButton>button {
        background-color: #00FFC2 !important;
        color: #000 !important;
        border: 4px solid #000 !important;
        border-radius: 0px !important;
        font-weight: 900 !important;
        text-transform: uppercase;
        box-shadow: 6px 6px 0px #FF00F5 !important;
        transition: all 0.1s ease;
    }

    .stButton>button:hover {
        transform: translate(-2px, -2px);
        box-shadow: 8px 8px 0px #FF00F5 !important;
    }

    .stButton>button:active {
        transform: translate(4px, 4px);
        box-shadow: 0px 0px 0px #FF00F5 !important;
    }

    /* Chat Messages */
    [data-testid="stChatMessage"] {
        background-color: rgba(0, 0, 0, 0.8) !important;
        border: 2px solid #FF00F5 !important;
        border-radius: 0px !important;
        box-shadow: 4px 4px 0px #00FFC2 !important;
        margin-bottom: 20px !important;
    }

    [data-testid="stChatMessageContent"] {
        color: #00FFC2 !important;
    }

    /* Input Field */
    .stChatInputContainer {
        border: 4px solid #FF00F5 !important;
        border-radius: 0px !important;
        background-color: #000 !important;
    }

    .stChatInputContainer textarea {
        color: #00FFC2 !important;
        font-family: 'Space Mono', monospace !important;
    }

    /* Success/Error/Warning Messages */
    .stAlert {
        border-radius: 0px !important;
        border: 3px solid #000 !important;
        box-shadow: 4px 4px 0px #000 !important;
    }

    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
    }
    ::-webkit-scrollbar-track {
        background: #000;
    }
    ::-webkit-scrollbar-thumb {
        background: #FF00F5;
        border: 3px solid #000;
    }

    </style>
    """, unsafe_allow_html=True)
