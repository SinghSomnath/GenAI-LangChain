"""
LangGraph Router Agent - AstraDB vs Web Crawl
FastAPI-powered REST API with PDF ingestion into AstraDB and live web crawling.

Python Version: 3.13.1

Install dependencies:
    pip install fastapi uvicorn pypdf langchain-text-splitters \
                duckduckgo-search beautifulsoup4 requests lxml

Run with:
    uvicorn langgraph_router_agent:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    POST /upload     - Upload a PDF and store its chunks into AstraDB
    POST /ask        - Submit a question (routed to AstraDB or Web Crawl)
    GET  /documents  - List all ingested PDF documents tracked in memory
    GET  /health     - Health check
    GET  /docs       - Auto-generated Swagger UI

Web-crawl flow (replaces Wikipedia):
    1. DuckDuckGo search  →  top N result URLs
    2. requests + BeautifulSoup  →  fetch & clean page text
    3. Concatenated content passed to LLM for answer generation
"""

from typing import Annotated, Literal, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

import os
import io
import re
import time
import hashlib
import logging
import requests
from datetime import datetime, timezone
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from duckduckgo_search import DDGS

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader

load_dotenv()


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

# Number of URLs DuckDuckGo returns for a web-crawl query
WEB_SEARCH_MAX_RESULTS: int = 5

# Max characters scraped from a single page (keeps LLM context manageable)
WEB_PAGE_MAX_CHARS: int = 4000

# HTTP request timeout in seconds
HTTP_TIMEOUT: int = 10

# User-agent sent with every crawl request
CRAWL_USER_AGENT: str = (
    "Mozilla/5.0 (compatible; LangGraphBot/1.0; +https://example.com/bot)"
)

# Domains that are unlikely to return useful plain text
BLOCKED_DOMAINS: frozenset[str] = frozenset({
    "youtube.com", "youtu.be", "twitter.com", "x.com",
    "instagram.com", "facebook.com", "tiktok.com",
    "linkedin.com", "reddit.com",
})


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="LangGraph Router Agent API",
    description=(
        "Upload PDFs into AstraDB and ask questions. "
        "The routing agent decides whether to answer from your uploaded "
        "documents (AstraDB) or by crawling live web pages."
    ),
    version="3.0.0",
)


# ============================================================================
# IN-MEMORY DOCUMENT REGISTRY
# ============================================================================

INGESTED_DOCS: dict[str, dict] = {}   # sha256 -> metadata dict


# ============================================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================================

class UploadResponse(BaseModel):
    filename: str
    file_sha256: str
    total_pages: int
    total_chunks: int
    chunk_size: int
    chunk_overlap: int
    collection: str
    ingested_at: str
    already_existed: bool = Field(
        description="True if this exact file was already ingested previously."
    )


class DocumentRecord(BaseModel):
    filename: str
    file_sha256: str
    total_pages: int
    total_chunks: int
    ingested_at: str


class DocumentListResponse(BaseModel):
    total: int
    documents: list[DocumentRecord]


class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="The question you want the agent to answer.",
        examples=["What are the latest advancements in quantum computing?"],
    )
    top_k: Optional[int] = Field(
        default=4,
        ge=1,
        le=10,
        description="Number of chunks to retrieve from AstraDB (ignored for web crawl).",
    )
    max_web_results: Optional[int] = Field(
        default=WEB_SEARCH_MAX_RESULTS,
        ge=1,
        le=10,
        description="How many web pages to crawl when the web-crawl path is chosen.",
    )


class CrawledPage(BaseModel):
    url: str
    title: str
    snippet: str


class RetrievedDocument(BaseModel):
    rank: int
    source: str
    content: str
    metadata: dict


class QuestionResponse(BaseModel):
    question: str
    route_decision: str = Field(description="ASTRADB or WEB_CRAWL")
    answer: str
    retrieved_documents: list[RetrievedDocument]
    crawled_pages: list[CrawledPage] = Field(
        default_factory=list,
        description="Pages fetched during a WEB_CRAWL turn (empty for ASTRADB turns).",
    )
    duration_seconds: float


class HealthResponse(BaseModel):
    status: str
    llm_model: str
    embedding_model: str
    astradb_collection: str
    ingested_documents: int


# ============================================================================
# STATE DEFINITION
# ============================================================================

class State(TypedDict):
    messages: Annotated[list, add_messages]
    route_decision: str          # "ASTRADB" | "WEB_CRAWL"
    retrieved_docs: list         # unified doc list for response generator
    crawled_pages: list          # [{url, title, snippet}] — web-crawl metadata
    top_k: int
    max_web_results: int


# ============================================================================
# SHARED SERVICES
# ============================================================================

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o",
    temperature=0,
)

embeddings = OpenAIEmbeddings(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="text-embedding-3-large",
    dimensions=1024,
)

astra_vector_store = AstraDBVectorStore(
    embedding=embeddings,
    collection_name=os.getenv("ASTRA_DB_COLLECTION", "default_collection"),
    api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
    token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
    namespace=os.getenv("ASTRA_DB_KEYSPACE"),
)


# ============================================================================
# PDF INGESTION HELPERS
# ============================================================================

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_text_from_pdf(raw_bytes: bytes) -> tuple[list[tuple[int, str]], int]:
    reader = PdfReader(io.BytesIO(raw_bytes))
    pages: list[tuple[int, str]] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            pages.append((i, text))
    return pages, len(reader.pages)


def chunk_pages(
    pages: list[tuple[int, str]],
    filename: str,
    file_sha256: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    documents: list[Document] = []
    for page_num, page_text in pages:
        chunks = splitter.split_text(page_text)
        for chunk_idx, chunk in enumerate(chunks, start=1):
            documents.append(Document(
                page_content=chunk,
                metadata={
                    "source": filename,
                    "file_sha256": file_sha256,
                    "page": page_num,
                    "chunk": chunk_idx,
                    "total_chunks_on_page": len(chunks),
                },
            ))
    return documents


# ============================================================================
# WEB CRAWL HELPERS
# ============================================================================

def _is_blocked(url: str) -> bool:
    """Return True if the URL's domain is in the blocked list."""
    try:
        host = urlparse(url).netloc.lower().lstrip("www.")
        return any(host == d or host.endswith("." + d) for d in BLOCKED_DOMAINS)
    except Exception:
        return False


def _clean_html(html: str) -> str:
    """
    Strip HTML tags and collapse whitespace.
    Tries to pull <article> / <main> first for cleaner text;
    falls back to <body>.
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove boilerplate tags
    for tag in soup(["script", "style", "nav", "header", "footer",
                     "aside", "form", "noscript", "iframe"]):
        tag.decompose()

    # Prefer semantic content containers
    container = (
        soup.find("article")
        or soup.find("main")
        or soup.find("div", {"id": re.compile(r"content|main|article", re.I)})
        or soup.body
    )

    raw_text = (container or soup).get_text(separator="\n")

    # Collapse blank lines / leading whitespace
    lines = [ln.strip() for ln in raw_text.splitlines() if ln.strip()]
    return "\n".join(lines)


def _page_title(html: str) -> str:
    """Extract <title> text from HTML."""
    soup = BeautifulSoup(html, "lxml")
    tag = soup.find("title")
    return tag.get_text(strip=True) if tag else "Untitled"


def search_and_crawl(query: str, max_results: int = WEB_SEARCH_MAX_RESULTS) -> list[dict]:
    """
    1. Run a DuckDuckGo text search for *query*.
    2. Filter blocked domains.
    3. Fetch each page with requests.
    4. Clean the HTML to plain text.
    5. Return a list of dicts: {url, title, content, snippet}.

    At most *max_results* pages are returned; pages that fail to fetch are skipped.
    """
    crawled: list[dict] = []

    # ── DuckDuckGo search ────────────────────────────────────────────────────
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results * 2))
    except Exception as exc:
        logger.error("DuckDuckGo search failed: %s", exc)
        return []

    logger.info("DuckDuckGo returned %d raw results for query: %s", len(results), query)

    for result in results:
        if len(crawled) >= max_results:
            break

        url: str = result.get("href", "")
        ddg_snippet: str = result.get("body", "")

        if not url or _is_blocked(url):
            logger.debug("Skipping blocked/empty URL: %s", url)
            continue

        # ── Fetch page ───────────────────────────────────────────────────────
        try:
            resp = requests.get(
                url,
                timeout=HTTP_TIMEOUT,
                headers={"User-Agent": CRAWL_USER_AGENT},
                allow_redirects=True,
            )
            resp.raise_for_status()
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                logger.debug("Skipping non-HTML response at %s (%s)", url, content_type)
                continue
        except Exception as exc:
            logger.warning("Failed to fetch %s: %s", url, exc)
            continue

        # ── Parse & clean ────────────────────────────────────────────────────
        try:
            title = _page_title(resp.text)
            full_text = _clean_html(resp.text)
        except Exception as exc:
            logger.warning("HTML parsing error for %s: %s", url, exc)
            continue

        if not full_text.strip():
            logger.debug("Empty text after cleaning for %s", url)
            continue

        # Truncate to keep LLM context manageable
        truncated = full_text[:WEB_PAGE_MAX_CHARS]

        crawled.append({
            "url": url,
            "title": title,
            "content": truncated,
            "snippet": ddg_snippet[:300],   # short preview for response metadata
        })
        logger.info("Crawled (%d/%d): %s", len(crawled), max_results, url)

    logger.info("Web crawl finished — %d pages collected", len(crawled))
    return crawled


# ============================================================================
# GRAPH NODES
# ============================================================================

def router_node(state: State) -> State:
    """Routes the question to either AstraDB or the web crawler."""
    user_question = state["messages"][0].content

    has_custom_docs = len(INGESTED_DOCS) > 0
    doc_hint = (
        f"Note: AstraDB currently holds {len(INGESTED_DOCS)} ingested PDF document(s). "
        "Prefer ASTRADB for questions that are likely answered by those files."
        if has_custom_docs
        else "Note: no PDFs have been uploaded yet — prefer WEB_CRAWL unless the question is clearly domain-specific."
    )

    routing_prompt = f"""You are a routing assistant. Decide whether the question below should be answered using:

1. **ASTRADB** – a vector database containing domain-specific PDF documents
   (company docs, technical manuals, research papers, internal reports, etc.)
2. **WEB_CRAWL** – live web search and page crawling for current or general information
   (recent news, public facts, product info, anything not in the uploaded docs)

{doc_hint}

Question: {user_question}

Respond with ONLY one of these two words: "ASTRADB" or "WEB_CRAWL".
"""

    response = llm.invoke([
        SystemMessage(content="You are a routing decision maker. Respond with only 'ASTRADB' or 'WEB_CRAWL'."),
        HumanMessage(content=routing_prompt),
    ])
    route_decision = response.content.strip().upper()

    if route_decision not in ["ASTRADB", "WEB_CRAWL"]:
        route_decision = "WEB_CRAWL"

    logger.info("Routing decision: %s", route_decision)
    return {
        "route_decision": route_decision,
        "messages": [AIMessage(content=f"Routing to {route_decision}...")],
    }


def astradb_retrieval_node(state: State) -> State:
    """Retrieves relevant chunks from AstraDB."""
    user_question = state["messages"][0].content
    top_k = state.get("top_k", 4)

    logger.info("Searching AstraDB — query: '%s'  top_k: %d", user_question, top_k)

    try:
        results = astra_vector_store.similarity_search(query=user_question, k=top_k)

        retrieved_docs = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "source": (
                    f"AstraDB | {doc.metadata.get('source', 'unknown')} "
                    f"p.{doc.metadata.get('page', '?')}"
                ),
                "rank": idx,
            }
            for idx, doc in enumerate(results, 1)
        ]

        logger.info("Retrieved %d chunks from AstraDB", len(retrieved_docs))
        return {
            "retrieved_docs": retrieved_docs,
            "crawled_pages": [],
            "messages": [AIMessage(content=f"Retrieved {len(retrieved_docs)} chunks from AstraDB")],
        }

    except Exception as exc:
        logger.error("AstraDB retrieval error: %s", exc)
        return {
            "retrieved_docs": [],
            "crawled_pages": [],
            "messages": [AIMessage(content=f"Error retrieving from AstraDB: {exc}")],
        }


def web_crawl_retrieval_node(state: State) -> State:
    """
    Runs a DuckDuckGo search, crawls the top result pages, and
    returns their cleaned text as retrieved documents.
    """
    user_question = state["messages"][0].content
    max_results = state.get("max_web_results", WEB_SEARCH_MAX_RESULTS)

    logger.info("Starting web crawl for: %s  (max_results=%d)", user_question, max_results)

    crawled = search_and_crawl(user_question, max_results=max_results)

    if not crawled:
        logger.warning("Web crawl returned no results for: %s", user_question)
        return {
            "retrieved_docs": [],
            "crawled_pages": [],
            "messages": [AIMessage(content="Web crawl returned no usable results.")],
        }

    # Convert crawled pages into the unified retrieved_docs format
    retrieved_docs = [
        {
            "content": page["content"],
            "metadata": {"url": page["url"], "title": page["title"]},
            "source": f"Web | {page['title']} ({page['url']})",
            "rank": idx,
        }
        for idx, page in enumerate(crawled, 1)
    ]

    # Lightweight metadata list for the API response
    crawled_pages_meta = [
        {"url": p["url"], "title": p["title"], "snippet": p["snippet"]}
        for p in crawled
    ]

    return {
        "retrieved_docs": retrieved_docs,
        "crawled_pages": crawled_pages_meta,
        "messages": [AIMessage(content=f"Crawled {len(crawled)} web pages")],
    }


def response_generator_node(state: State) -> State:
    """Generates the final answer from retrieved context."""
    user_question = state["messages"][0].content
    retrieved_docs = state["retrieved_docs"]
    route_decision = state.get("route_decision", "")

    if not retrieved_docs:
        hint = (
            "Try uploading a PDF via /upload."
            if route_decision == "ASTRADB"
            else "The web crawl did not return usable content — try rephrasing."
        )
        return {
            "messages": [
                AIMessage(content=(
                    f"I couldn't retrieve any relevant information for your question. {hint}"
                ))
            ]
        }

    context = "\n---\n".join(
        f"[{doc['source']} | Rank {doc['rank']}]\n{doc['content']}"
        for doc in retrieved_docs
    )

    source_instruction = (
        "Cite the filename and page number when referencing AstraDB content."
        if route_decision == "ASTRADB"
        else "Cite the page title and URL when referencing web content."
    )

    response_prompt = f"""Answer the user's question using only the retrieved information below.

User Question: {user_question}

Retrieved Information:
{context}

Instructions:
- Give a clear, accurate answer grounded in the retrieved information.
- {source_instruction}
- If the content doesn't fully answer the question, say so explicitly.
- Be concise but complete.

Answer:"""

    response = llm.invoke([
        SystemMessage(content="You are a helpful assistant that answers questions from retrieved sources."),
        HumanMessage(content=response_prompt),
    ])
    logger.info("Final response generated")
    return {"messages": [AIMessage(content=response.content)]}


# ============================================================================
# CONDITIONAL EDGE
# ============================================================================

def route_to_retrieval(state: State) -> Literal["astradb_retrieval", "web_crawl_retrieval"]:
    return "astradb_retrieval" if state.get("route_decision") == "ASTRADB" else "web_crawl_retrieval"


# ============================================================================
# GRAPH BUILDER
# ============================================================================

def build_graph():
    graph = StateGraph(State)
    graph.add_node("router", router_node)
    graph.add_node("astradb_retrieval", astradb_retrieval_node)
    graph.add_node("web_crawl_retrieval", web_crawl_retrieval_node)
    graph.add_node("response_generator", response_generator_node)

    graph.set_entry_point("router")
    graph.add_conditional_edges(
        "router",
        route_to_retrieval,
        {
            "astradb_retrieval": "astradb_retrieval",
            "web_crawl_retrieval": "web_crawl_retrieval",
        },
    )
    graph.add_edge("astradb_retrieval", "response_generator")
    graph.add_edge("web_crawl_retrieval", "response_generator")
    graph.add_edge("response_generator", "__end__")
    return graph.compile()


AGENT = build_graph()


# ============================================================================
# FASTAPI ENDPOINTS
# ============================================================================

@app.post(
    "/upload",
    response_model=UploadResponse,
    tags=["Ingestion"],
    summary="Upload a PDF and store its chunks into AstraDB",
)
async def upload_pdf(
    file: UploadFile = File(..., description="PDF file to ingest"),
    chunk_size: int = Form(
        default=1000, ge=100, le=5000,
        description="Max characters per chunk (default 1000)",
    ),
    chunk_overlap: int = Form(
        default=200, ge=0, le=1000,
        description="Character overlap between consecutive chunks (default 200)",
    ),
):
    """
    Upload a **PDF file**. The endpoint will:

    1. Validate the file is a PDF.
    2. Compute a **SHA-256 fingerprint** to skip duplicate uploads.
    3. Extract text page-by-page with **pypdf**.
    4. Split into overlapping chunks with **RecursiveCharacterTextSplitter**.
    5. Embed and store every chunk in **AstraDB** with source + page metadata.

    **Example curl**
    ```bash
    curl -X POST http://localhost:8000/upload \\
      -F "file=@company_handbook.pdf" \\
      -F "chunk_size=1000" \\
      -F "chunk_overlap=200"
    ```
    """
    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Please upload a PDF.",
        )
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="File must have a .pdf extension.")

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    file_hash = _sha256(raw_bytes)
    logger.info("Received '%s'  size=%d bytes  sha256=%s", file.filename, len(raw_bytes), file_hash)

    if file_hash in INGESTED_DOCS:
        existing = INGESTED_DOCS[file_hash]
        logger.info("Duplicate upload — skipping ingestion for '%s'", file.filename)
        return UploadResponse(
            **existing,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            collection=os.getenv("ASTRA_DB_COLLECTION", "default_collection"),
            already_existed=True,
        )

    try:
        pages, total_pages = extract_text_from_pdf(raw_bytes)
    except Exception as exc:
        logger.error("PDF parsing error: %s", exc)
        raise HTTPException(status_code=422, detail=f"Could not parse PDF: {exc}")

    if not pages:
        raise HTTPException(
            status_code=422,
            detail=(
                "No extractable text found in the PDF. "
                "It may be a scanned image — please use an OCR-processed PDF."
            ),
        )

    logger.info("Extracted text from %d/%d pages", len(pages), total_pages)

    documents = chunk_pages(
        pages=pages,
        filename=file.filename,
        file_sha256=file_hash,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    logger.info("Created %d chunks", len(documents))

    try:
        astra_vector_store.add_documents(documents)
        logger.info("Stored %d chunks into AstraDB collection '%s'",
                    len(documents), os.getenv("ASTRA_DB_COLLECTION", "default_collection"))
    except Exception as exc:
        logger.error("AstraDB write error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to store chunks in AstraDB: {exc}")

    record = {
        "filename": file.filename,
        "file_sha256": file_hash,
        "total_pages": total_pages,
        "total_chunks": len(documents),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }
    INGESTED_DOCS[file_hash] = record

    return UploadResponse(
        **record,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        collection=os.getenv("ASTRA_DB_COLLECTION", "default_collection"),
        already_existed=False,
    )


@app.get(
    "/documents",
    response_model=DocumentListResponse,
    tags=["Ingestion"],
    summary="List all PDFs ingested this session",
)
async def list_documents():
    """Returns metadata for every PDF ingested since the server started."""
    records = [DocumentRecord(**v) for v in INGESTED_DOCS.values()]
    return DocumentListResponse(total=len(records), documents=records)


@app.post(
    "/ask",
    response_model=QuestionResponse,
    tags=["Agent"],
    summary="Ask a question — routed to AstraDB or web crawl automatically",
)
async def ask_question(payload: QuestionRequest):
    """
    Submit a question. The agent will:

    1. **Route** — decide between AstraDB (uploaded PDFs) and live web crawling.
    2. **Retrieve** — either fetch AstraDB chunks or search + crawl web pages.
    3. **Generate** — produce a grounded answer with source citations.

    When web crawl is chosen, the response includes a `crawled_pages` list
    with the URL, title, and a short snippet for every page that was fetched.

    **Request body (JSON)**
    ```json
    {
      "question": "What are the latest advancements in quantum computing?",
      "top_k": 4,
      "max_web_results": 5
    }
    ```
    """
    start = time.perf_counter()
    logger.info("Received question: %s", payload.question)

    try:
        final_state = AGENT.invoke({
            "messages": [HumanMessage(content=payload.question)],
            "route_decision": "",
            "retrieved_docs": [],
            "crawled_pages": [],
            "top_k": payload.top_k,
            "max_web_results": payload.max_web_results,
        })
    except Exception as exc:
        logger.error("Agent error: %s", exc)
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {exc}")

    duration = round(time.perf_counter() - start, 3)
    answer = final_state["messages"][-1].content

    retrieved_documents = [
        RetrievedDocument(
            rank=doc["rank"],
            source=doc["source"],
            content=doc["content"],
            metadata=doc.get("metadata", {}),
        )
        for doc in final_state.get("retrieved_docs", [])
    ]

    crawled_pages = [
        CrawledPage(**p) for p in final_state.get("crawled_pages", [])
    ]

    return QuestionResponse(
        question=payload.question,
        route_decision=final_state.get("route_decision", "UNKNOWN"),
        answer=answer,
        retrieved_documents=retrieved_documents,
        crawled_pages=crawled_pages,
        duration_seconds=duration,
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Utility"],
    summary="Service health check",
)
async def health_check():
    """Returns configuration info without making any external API calls."""
    return HealthResponse(
        status="ok",
        llm_model="gpt-4o",
        embedding_model="text-embedding-3-large",
        astradb_collection=os.getenv("ASTRA_DB_COLLECTION", "default_collection"),
        ingested_documents=len(INGESTED_DOCS),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected error occurred. Check server logs for details."},
    )


# ============================================================================
# ENTRYPOINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    import threading
      
    def run_fastapi():
        uvicorn.run(app, host="0.0.0.0", port=8000)
    
    fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
    fastapi_thread.start()