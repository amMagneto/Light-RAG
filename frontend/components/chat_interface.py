import streamlit as st
from .api_utils import get_api_response


def _render_citations(citations):
    if not citations:
        return
    st.caption("Sources")
    for citation in citations:
        section = " > ".join(citation.get("section_path") or []) or "document"
        page = f" · p. {citation['page_number']}" if citation.get("page_number") else ""
        st.markdown(
            f"**[{citation['index']}]** {citation.get('document_name') or 'unknown'} — {section}{page} "
            f"`rel {citation.get('relevance_score', 0):.2f}`"
        )


def _render_trace(trace):
    if not trace:
        return
    stages = trace.get("stage_latencies_ms") or {}
    stage_text = " · ".join(f"{name}: {ms:.0f}ms" for name, ms in stages.items())
    st.caption(
        f"trace `{trace.get('run_id', '')[:8]}` | intent: **{trace.get('intent')}** | "
        f"verdict: **{trace.get('verdict')}** | attempts: {trace.get('attempts', 1)} | {stage_text}"
    )


def render_chat_interface():
    st.title("💬 Light-RAG")
    st.caption("Vertical retrieval platform — grounded answers with citations")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = None

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                _render_citations(message.get("citations"))
                _render_trace(message.get("trace"))

    if prompt := st.chat_input("Ask your knowledge base..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving → grading → generating → verifying..."):
                response = get_api_response(prompt, st.session_state.session_id)

                if response:
                    answer = response["answer"]
                    st.session_state.session_id = response["session_id"]
                    confidence = response.get("confidence")
                    st.markdown(answer)
                    if confidence is not None:
                        st.progress(min(1.0, max(0.0, confidence)), text=f"confidence {confidence:.2f}")
                    _render_citations(response.get("citations"))
                    if response.get("uncertainty"):
                        st.warning(f"Uncertainty: {response['uncertainty']}")
                    _render_trace(response.get("trace"))
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "citations": response.get("citations"),
                            "trace": response.get("trace"),
                        }
                    )
                else:
                    st.error("Failed to reach the retrieval engine.")
