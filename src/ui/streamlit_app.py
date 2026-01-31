# src/ui/streamlit_app.py

from __future__ import annotations

import os
import requests
import streamlit as st

DEFAULT_API_URL = "http://localhost:8000/api/chat"
API_URL = os.getenv("RENEWAL_API_URL", DEFAULT_API_URL)


def call_backend(user_query: str, membership_nbr: int | None) -> dict:
    """Call FastAPI /api/chat endpoint and return parsed JSON."""
    payload: dict = {"user_query": user_query}
    if membership_nbr is not None:
        payload["membership_nbr"] = membership_nbr

    resp = requests.post(API_URL, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _init_session_state() -> None:
    if "messages" not in st.session_state:
        # list[dict]: {"role": "user"/"assistant", "content": str}
        st.session_state["messages"] = []


def main() -> None:
    st.set_page_config(
        page_title="Renewal Intelligence Assistant",
        page_icon="💬",
        layout="wide",
    )

    _init_session_state()

    with st.sidebar:
        st.title("Settings")
        st.markdown(f"**Backend URL**: `{API_URL}`")
        member_id_input = st.text_input(
            "Membership Number (optional)",
            value="",
            help="If provided, the backend can personalize NBA using this member.",
        )
        if st.button("Clear chat"):
            st.session_state["messages"] = []

    st.title("Agentic Renewal Intelligence Assistant")
    st.write(
        "Ask questions about renewal drivers, personas, and next-best-actions. "
        "Example: *“What are the best actions for persona 3 in first year?”*"
    )

    # Chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # User input
    user_query = st.chat_input("Type your question here...")

    if user_query:
        # Append user message
        st.session_state["messages"].append({"role": "user", "content": user_query})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_query)

        # Parse membership_nbr
        membership_nbr: int | None = None
        if member_id_input.strip():
            try:
                membership_nbr = int(member_id_input.strip())
            except ValueError:
                st.warning("Membership number must be an integer. Ignoring it for this query.")

        # Call backend
        try:
            backend_resp = call_backend(user_query, membership_nbr)
            answer = backend_resp.get("answer", "").strip() or "(No answer returned)"
            intent = backend_resp.get("intent", "unknown")
            used_member = backend_resp.get("membership_nbr")

            # Store only the answer in history
            st.session_state["messages"].append(
                {"role": "assistant", "content": answer}
            )

            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    with st.expander("Debug info (intent, member, etc.)", expanded=False):
                        st.write(
                            {
                                "intent": intent,
                                "membership_nbr": used_member,
                            }
                        )

        except requests.HTTPError as e:
            msg = f"Backend error: {e.response.status_code} {e.response.text}"
            st.error(msg)
        except Exception as e:
            st.error(f"Unexpected error calling backend: {e}")


if __name__ == "__main__":
    main()
