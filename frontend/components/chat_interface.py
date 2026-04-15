import streamlit as st
from .api_utils import get_api_response

def render_chat_interface():
    st.title("💬 ChainReaction")
    st.caption("Conversational RAG powered by First Principles")

    # 1. Initialize session-based memory if empty
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    # 2. Display existing history (The 'Persistent' loop)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 3. Handle User Input (The Ingestion/Query step)
    if prompt := st.chat_input("Ask a first principles question..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 4. Call Backend (The Brain/Retrieval step)
        with st.chat_message("assistant"):
            with st.spinner("Thinking from bedrock..."):
                response = get_api_response(prompt, st.session_state.session_id)
                
                if response:
                    answer = response["answer"]
                    # Update session_id from backend to maintain the loop
                    st.session_state.session_id = response["session_id"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error("Failed to reach the RAG engine.")