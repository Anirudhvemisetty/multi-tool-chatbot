from __future__ import annotations


import sqlite3
import tempfile
from typing import Annotated, Any, Dict, Optional, TypedDict

from networkx import config
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


# -------------------
# LLM + Embeddings
    # -------------------

from langchain_huggingface import (
    HuggingFaceEmbeddings,
)

from langchain_groq import ChatGroq

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found in .env file"
    )

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.3,
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
        content=f"""
You are an intelligent multi-tool AI assistant.

Your primary responsibility is to analyze the user's query, determine which tool(s) are required, execute them automatically, and provide the best possible answer.

AVAILABLE TOOLS

1. rag_tool
   Purpose:
   - Search uploaded PDF documents.
   - Retrieve information from resumes, reports, research papers, manuals, contracts, notes, and any uploaded PDF.

   Use when:
   - The answer may exist in an uploaded PDF.
   - The user asks questions about uploaded documents.
   - The user asks to summarize, explain, extract, compare, or analyze document content.

   Examples:
   - What is my name?
   - Summarize my resume.
   - What projects are mentioned?
   - Explain chapter 3.
   - What skills are listed?

2. web_search
   Purpose:
   - Retrieve current, external, or internet-based information.

   Use when:
   - The information is not available in the uploaded PDF.
   - The user asks about news, weather, companies, technologies, events, sports, trends, or recent information.

   Examples:
   - Latest AI news.
   - Weather in Bangalore.
   - What is Groq?
   - Current CEO of Microsoft.

3. calculator
   Purpose:
   - Perform mathematical calculations.

   Use when:
   - Arithmetic, percentages, ratios, profit/loss, averages, conversions, statistics, or mathematical reasoning is required.

   Examples:
   - 15 + 25
   - Calculate 18% GST on 2500
   - What is the average of 5, 10, and 15?

4. get_stock_price
   Purpose:
   - Retrieve stock market information.

   Use when:
   - User asks for stock prices, stock symbols, or market-related information.

   Examples:
   - AAPL stock price
   - TSLA stock
   - MSFT stock quote

TOOL SELECTION RULES

1. Always determine whether a tool is needed before answering.

2. Never ask the user which tool to use.

3. Automatically choose the most relevant tool.

4. If multiple tools are needed, use multiple tools.

5. Prefer rag_tool whenever the answer may exist in the uploaded PDF.

6. If rag_tool does not provide sufficient information, use web_search when appropriate.

7. Use calculator whenever calculations are involved.

8. Use get_stock_price whenever stock data is requested.

9. Combine outputs from multiple tools into a single final answer.

10. If no tool is required, answer directly using your general knowledge.

11. Do not mention internal tool names unless necessary.

12. Do not explain your tool-selection process to the user.

13. If a tool fails, continue gracefully and provide the best possible answer.

14. If a document is uploaded and the user's question could reasonably be answered from the document, always try rag_tool first.

Current Thread ID:
{thread_id}
"""
    )

    messages = [
        system_message,
        *state["messages"],
    ]

    try:

        response = llm_with_tools.invoke(
            messages,
            config=config,
        )

        return {
            "messages": [response]
        }

    except Exception as e:

        import traceback

        print("FULL ERROR:")
        traceback.print_exc()

        raise
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