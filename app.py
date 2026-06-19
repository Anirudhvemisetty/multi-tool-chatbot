import uuid

import streamlit as st
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    ToolMessage,
)

from langraph_rag_backend import (
    chatbot,
    ingest_pdf,
    retrieve_all_threads,
    thread_document_metadata,
)


# =====================================================
# Utility Functions
# =====================================================

def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def reset_chat():
    new_thread = generate_thread_id()

    st.session_state["thread_id"] = new_thread
    st.session_state["message_history"] = []

    add_thread(new_thread)


def load_conversation(thread_id):
    try:
        state = chatbot.get_state(
            config={
                "configurable": {
                    "thread_id": thread_id
                }
            }
        )

        return state.values.get("messages", [])

    except Exception:
        return []


# =====================================================
# Session State Initialization
# =====================================================

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "ingested_docs" not in st.session_state:
    st.session_state["ingested_docs"] = {}

add_thread(st.session_state["thread_id"])

thread_id = st.session_state["thread_id"]
thread_key = str(thread_id)

thread_docs = st.session_state["ingested_docs"].setdefault(
    thread_key,
    {},
)

threads = st.session_state["chat_threads"][::-1]

selected_thread = None

# =====================================================
# Sidebar
# =====================================================

st.sidebar.title("📄 LangGraph PDF Chatbot")

st.sidebar.markdown(
    f"**Current Thread**\n\n`{thread_key}`"
)

if st.sidebar.button(
    "➕ New Chat",
    use_container_width=True,
):
    reset_chat()
    st.rerun()

# =====================================================
# PDF Upload
# =====================================================

st.sidebar.subheader("Upload PDF")

uploaded_pdf = st.sidebar.file_uploader(
    "Choose a PDF",
    type=["pdf"],
)

if uploaded_pdf is not None:

    if uploaded_pdf.name in thread_docs:

        st.sidebar.info(
            f"{uploaded_pdf.name} already indexed."
        )

    else:

        with st.sidebar.status(
            "Indexing PDF...",
            expanded=True,
        ) as status_box:

            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name,
            )

            thread_docs[uploaded_pdf.name] = summary

            status_box.update(
                label="✅ PDF Indexed",
                state="complete",
                expanded=False,
            )

# =====================================================
# Show Active Document
# =====================================================

if thread_docs:

    latest_doc = list(thread_docs.values())[-1]

    st.sidebar.success(
        f"""
PDF: {latest_doc.get("filename")}

Pages: {latest_doc.get("documents")}

Chunks: {latest_doc.get("chunks")}
"""
    )

else:
    st.sidebar.info(
        "No PDF uploaded for this thread."
    )

# =====================================================
# Past Conversations
# =====================================================

st.sidebar.divider()
st.sidebar.subheader("Past Conversations")

if not threads:

    st.sidebar.write(
        "No conversations found."
    )

else:

    for tid in threads:

        if st.sidebar.button(
            str(tid),
            key=f"thread_{tid}",
            use_container_width=True,
        ):
            selected_thread = tid

# =====================================================
# Main UI
# =====================================================

st.title("🤖 Multi Utility Chatbot")

st.caption(
    "PDF RAG + Web Search + Calculator + Tools"
)

# =====================================================
# Display Chat History
# =====================================================

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# =====================================================
# User Input
# =====================================================

user_input = st.chat_input(
    "Ask a question..."
)

if user_input:

    # Show User Message
    st.session_state["message_history"].append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    CONFIG = {
        "configurable": {
            "thread_id": thread_key
        }
    }

    # Assistant Response
    with st.chat_message("assistant"):

        status_holder = {
            "box": None
        }

        def stream_response():

            for chunk, metadata in chatbot.stream(
                {
                    "messages": [
                        HumanMessage(
                            content=user_input
                        )
                    ]
                },
                config=CONFIG,
                stream_mode="messages",
            ):

                # Tool Calls
                if isinstance(
                    chunk,
                    ToolMessage,
                ):

                    tool_name = getattr(
                        chunk,
                        "name",
                        "tool",
                    )

                    if status_holder["box"] is None:

                        status_holder["box"] = st.status(
                            f"🔧 Running `{tool_name}`...",
                            expanded=True,
                        )

                    else:

                        status_holder["box"].update(
                            label=f"🔧 Running `{tool_name}`...",
                            state="running",
                            expanded=True,
                        )

                # AI Output
                if isinstance(
                    chunk,
                    AIMessage,
                ):

                    if chunk.content:
                        yield chunk.content

        assistant_response = st.write_stream(
            stream_response()
        )

        if status_holder["box"]:

            status_holder["box"].update(
                label="✅ Tool Execution Complete",
                state="complete",
                expanded=False,
            )

    st.session_state["message_history"].append(
        {
            "role": "assistant",
            "content": assistant_response,
        }
    )

    # Document Info
    doc_meta = thread_document_metadata(
        thread_key
    )

    if doc_meta:

        st.caption(
            f"""
Indexed Document:
{doc_meta.get("filename")}
|
Pages: {doc_meta.get("documents")}
|
Chunks: {doc_meta.get("chunks")}
"""
        )

# =====================================================
# Switch Conversation
# =====================================================

if selected_thread:

    st.session_state["thread_id"] = selected_thread

    messages = load_conversation(
        selected_thread
    )

    history = []

    for msg in messages:

        if isinstance(
            msg,
            HumanMessage,
        ):

            history.append(
                {
                    "role": "user",
                    "content": msg.content,
                }
            )

        elif isinstance(
            msg,
            AIMessage,
        ):

            history.append(
                {
                    "role": "assistant",
                    "content": msg.content,
                }
            )

    st.session_state[
        "message_history"
    ] = history

    st.session_state[
        "ingested_docs"
    ].setdefault(
        str(selected_thread),
        {},
    )

    st.rerun()