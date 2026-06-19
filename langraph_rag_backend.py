from __future__ import annotations


import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

import requests


from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.tools import DuckDuckGoSearchRun

from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.tools import tool


from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

# -------------------
# Load Environment
# -------------------
from dotenv import load_dotenv
import os

from dotenv import load_dotenv
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))

print("TOKEN FOUND:", os.getenv("HUGGINGFACEHUB_API_TOKEN"))

HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not HF_TOKEN:
    raise ValueError("HUGGINGFACEHUB_API_TOKEN not found in .env file")
# -------------------
# LLM + Embeddings
    # -------------------

from langchain_huggingface import (
    HuggingFaceEmbeddings,
    HuggingFaceEndpoint,
    ChatHuggingFace,
)

llm_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    huggingfacehub_api_token=HF_TOKEN,
    temperature=0.3,
    max_new_tokens=1024,
)

llm = ChatHuggingFace(
    llm=llm_endpoint
)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)
# -------------------
# Storage
# -------------------

_THREAD_RETRIEVERS: Dict[str, Any] = {}
_THREAD_METADATA: Dict[str, dict] = {}

CURRENT_THREAD_ID = None

# -------------------
# Retriever Helpers
# -------------------
def _get_retriever(thread_id: Optional[str]):
    if thread_id and str(thread_id) in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[str(thread_id)]
    return None


def ingest_pdf(
    file_bytes: bytes,
    thread_id: str,
    filename: Optional[str] = None,
) -> dict:
    """
    Create FAISS vector store from uploaded PDF
    """

    if not file_bytes:
        raise ValueError("No PDF bytes provided.")

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""],
        )

        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(
            chunks,
            embeddings,
        )

        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4},
        )

        _THREAD_RETRIEVERS[str(thread_id)] = retriever

        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

        return {
            "filename": filename or os.path.basename(temp_path),
            "documents": len(docs),
            "chunks": len(chunks),
        }

    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -------------------
# Search Tool
# -------------------
duckduckgo = DuckDuckGoSearchRun(region="us-en")


@tool
def web_search(query: str) -> str:
    """
    Search the web using DuckDuckGo.
    """
    try:
        return duckduckgo.run(query)
    except Exception as e:
        return f"Search error: {str(e)}"


# -------------------
# Calculator Tool
# -------------------
@tool
def calculator(
    first_num: float,
    second_num: float,
    operation: str,
) -> dict:
    """
    Basic calculator.
    Supported operations:
    add, sub, mul, div
    """

    try:
        if operation == "add":
            result = first_num + second_num

        elif operation == "sub":
            result = first_num - second_num

        elif operation == "mul":
            result = first_num * second_num

        elif operation == "div":

            if second_num == 0:
                return {
                    "error": "Division by zero not allowed."
                }

            result = first_num / second_num

        else:
            return {
                "error": f"Unsupported operation '{operation}'"
            }

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result,
        }

    except Exception as e:
        return {"error": str(e)}


# -------------------
# Stock Tool
# -------------------
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Get stock price using Alpha Vantage.
    Example:
    AAPL
    TSLA
    MSFT
    """

    try:
        api_key = os.getenv(
            "ALPHA_VANTAGE_API_KEY",
            "demo"
        )

        url = (
            "https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE"
            f"&symbol={symbol}"
            f"&apikey={api_key}"
        )

        response = requests.get(
            url,
            timeout=15,
        )

        response.raise_for_status()

        return response.json()

    except Exception as e:
        return {
            "symbol": symbol,
            "error": str(e),
        }

@tool
def rag_tool(query: str) -> str:
    """
    Search the uploaded PDF and return relevant content.
    """

    global CURRENT_THREAD_ID

    print(f"RAG THREAD: {CURRENT_THREAD_ID}")
    print(f"AVAILABLE THREADS: {list(_THREAD_RETRIEVERS.keys())}")

    retriever = _get_retriever(CURRENT_THREAD_ID)

    if retriever is None:
        return "No PDF has been uploaded for this chat."

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found in the uploaded PDF."

    context = "\n\n".join(
        [doc.page_content for doc in docs]
    )

    source_file = _THREAD_METADATA.get(
        str(CURRENT_THREAD_ID),
        {}
    ).get(
        "filename",
        "Unknown File"
    )

    return f"""
Source File: {source_file}

Relevant Content:

{context}
"""

# -------------------
# Tools
# -------------------
tools = [
    web_search,
    calculator,
    get_stock_price,
    rag_tool,
]

llm_with_tools = llm.bind_tools(tools)

# -------------------
# State
# -------------------
class ChatState(TypedDict):
    messages: Annotated[
        list[BaseMessage],
        add_messages,
    ]


# -------------------
# Chat Node
# -------------------
def chat_node(
    state: ChatState,
    config=None,
):
    """
    Main LLM node.
    """

    global CURRENT_THREAD_ID

    thread_id = None

    if config:
        thread_id = (
            config.get(
                "configurable",
                {}
            ).get("thread_id")
        )

    # Store current thread globally for rag_tool
    CURRENT_THREAD_ID = thread_id

    system_message = SystemMessage(
        content=f"""You are an intelligent AI assistant with access to multiple tools.

Available Tools:

1. rag_tool

   * Retrieves information from uploaded PDF documents.
   * Use whenever information may exist inside the uploaded document.
   * Examples:

     * "What is my name?"
     * "What is my school name?"
     * "Summarize my resume."
     * "What projects have I done?"
     * Any question that could be answered from the uploaded PDF.

2. web_search

   * Searches the internet for current or external information.
   * Use when information is not available in the PDF or requires live data.
   * Examples:

     * Weather
     * News
     * Company information
     * Current events

3. calculator

   * Performs mathematical calculations.
   * Use whenever numerical computation is required.

4. get_stock_price

   * Retrieves stock market data.

Decision Process:

* First determine where the answer is most likely located.
* If the answer can be found in the uploaded PDF, use rag_tool.
* If the answer requires current or external information, use web_search.
* If calculations are needed, use calculator.
* If both document data and web information are required, use both tools.
* Do not ask the user which tool to use.
* Select tools automatically.
* Combine information from multiple tools when needed.

Current Thread ID:
{thread_id}

"""
    )

    messages = [
        system_message,
        *state["messages"],
    ]

    response = llm_with_tools.invoke(
        messages,
        config=config,
    )

    return {
        "messages": [response]
    }

# -------------------
# Tool Node
# -------------------
tool_node = ToolNode(tools)

# -------------------
# SQLite Checkpointer
# -------------------
conn = sqlite3.connect(
    "chatbot.db",
    check_same_thread=False,
)

checkpointer = SqliteSaver(conn)

# -------------------
# Graph
# -------------------
graph = StateGraph(ChatState)

graph.add_node(
    "chat_node",
    chat_node,
)

graph.add_node(
    "tools",
    tool_node,
)

graph.add_edge(
    START,
    "chat_node",
)

graph.add_conditional_edges(
    "chat_node",
    tools_condition,
)

graph.add_edge(
    "tools",
    "chat_node",
)

chatbot = graph.compile(
    checkpointer=checkpointer
)

# -------------------
# Utility Functions
# -------------------
def retrieve_all_threads():
    """
    Return all thread ids.
    """

    thread_ids = set()

    for checkpoint in checkpointer.list(None):

        try:
            thread_id = checkpoint.config[
                "configurable"
            ]["thread_id"]

            thread_ids.add(thread_id)

        except Exception:
            pass

    return list(thread_ids)


def thread_has_document(
    thread_id: str,
) -> bool:
    return (
        str(thread_id)
        in _THREAD_RETRIEVERS
    )


def thread_document_metadata(
    thread_id: str,
) -> dict:
    return _THREAD_METADATA.get(
        str(thread_id),
        {},
    )


# -------------------
# Example Usage
# -------------------
if __name__ == "__main__":

    config = {
        "configurable": {
            "thread_id": "demo-thread"
        }
    }

    while True:

        query = input("\nUser: ")

        if query.lower() in {
            "exit",
            "quit",
        }:
            break

        result = chatbot.invoke(
            {
                "messages": [
                    ("user", query)
                ]
            },
            config=config,
        )

        print(
            "\nAssistant:",
            result["messages"][-1].content,
        )