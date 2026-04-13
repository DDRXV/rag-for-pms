"""
Helper layer for the RAG for PMs notebook series.

This module hides the boring plumbing so every lesson cell in every chapter
can stay three to ten lines long. The notebooks import a handful of named
functions from here. Students are not expected to read this file during the
lessons, but it is a short single file by design so anyone curious can open
it later and see exactly what is happening.
"""

import os
from pathlib import Path

# Silence chromadb telemetry in case any stale chromadb install is still
# present in the runtime. We do not use chromadb, but its telemetry module
# fires on import and spams the notebook with posthog.capture errors when
# the installed chromadb version is out of step with the installed posthog
# version. Setting this env var before any import that might transitively
# pull chromadb prevents those messages from ever appearing.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "False")

import pandas as pd
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-4o-mini"


def get_keys() -> None:
    """
    Pull OPENAI_API_KEY, LANGCHAIN_API_KEY, and COHERE_API_KEY from Colab
    secrets. Fall back to existing os.environ values when not running in
    Colab. Also enable LangSmith tracing for the rest of the session.
    """
    try:
        from google.colab import userdata  # type: ignore

        def _fetch(name: str) -> str | None:
            try:
                return userdata.get(name)
            except Exception:
                return None

        openai_key = _fetch("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
        langchain_key = _fetch("LANGCHAIN_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
        cohere_key = _fetch("COHERE_API_KEY") or os.environ.get("COHERE_API_KEY")
    except ImportError:
        openai_key = os.environ.get("OPENAI_API_KEY")
        langchain_key = os.environ.get("LANGCHAIN_API_KEY")
        cohere_key = os.environ.get("COHERE_API_KEY")

    if not openai_key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Add it to Colab secrets (key icon on "
            "the left sidebar) or set it in os.environ before calling get_keys."
        )

    os.environ["OPENAI_API_KEY"] = openai_key

    if langchain_key:
        os.environ["LANGCHAIN_API_KEY"] = langchain_key
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = "rag-for-pms"
        tracing_status = "LangSmith tracing enabled."
    else:
        os.environ.pop("LANGCHAIN_TRACING_V2", None)
        os.environ.pop("LANGCHAIN_API_KEY", None)
        print("LANGCHAIN_API_KEY is missing. LangSmith traces will be skipped for this run.")
        tracing_status = "LangSmith tracing skipped."

    if cohere_key:
        os.environ["COHERE_API_KEY"] = cohere_key
    else:
        print("COHERE_API_KEY is missing. Chapters that use Cohere rerank will fail until you add it.")

    print(f"Keys loaded. {tracing_status}")


def load_corpus(data_dir: str = "data/skillagents") -> list:
    """
    Load every .pdf and .md file from data_dir as LangChain Documents.
    PDFs go through PyPDFLoader. Markdown files go through TextLoader.
    Each document has metadata['source'] set to the filename only.
    """
    path = Path(data_dir)
    if not path.is_absolute():
        path = Path.cwd() / path

    if not path.exists():
        raise FileNotFoundError(
            f"Corpus directory not found: {path}. "
            "Make sure you have cloned the repo and changed into its root."
        )

    docs: list = []
    for file in sorted(path.iterdir()):
        if file.suffix == ".pdf":
            pages = PyPDFLoader(str(file)).load()
            combined_text = "\n\n".join(p.page_content for p in pages)
            docs.append(
                Document(
                    page_content=combined_text,
                    metadata={"source": file.name},
                )
            )
        elif file.suffix == ".md":
            loaded = TextLoader(str(file), encoding="utf-8").load()
            for d in loaded:
                d.metadata["source"] = file.name
            docs.extend(loaded)

    print(f"Loaded {len(docs)} documents from {path.name}")
    return docs


def make_chunks(
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    data_dir: str = "data/skillagents",
) -> list:
    """
    Load the corpus and split it into chunks using RecursiveCharacterTextSplitter.
    Returns the list of chunk Documents without building any index. Useful for
    inspecting what the splitter actually produces before you embed and index.
    """
    docs = load_corpus(data_dir)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)


def build_index(
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    data_dir: str = "data/skillagents",
):
    """
    Load the corpus, split it into chunks, embed each chunk with OpenAI,
    and index the result in a fresh in-memory FAISS vector store. Returns
    the vector store. Every call creates a new store, so students can call
    build_index repeatedly in the same notebook without stale state.
    """
    chunks = make_chunks(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        data_dir=data_dir,
    )

    embeddings = OpenAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL)
    vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)

    print(
        f"Indexed {len(chunks)} chunks "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
    )
    return vectorstore


def search(index, question: str, k: int = 5) -> list:
    """
    Run the question against the vector store and return the raw list of
    (Document, distance) pairs from FAISS. Lower distance means a closer
    match under L2 similarity.
    """
    return index.similarity_search_with_score(question, k=k)


def _clean_preview(text: str, max_chars: int) -> str:
    preview = " ".join(text.split())
    if len(preview) > max_chars:
        preview = preview[: max_chars - 1] + "\u2026"
    return preview


def show_results(results, question: str | None = None, max_chars: int = 200):
    """
    Turn a list of (Document, distance) pairs into a styled pandas table.
    Columns: rank, source, score, preview. The score column uses a green
    to red gradient where green is the closest match (lowest L2 distance).
    """
    if question:
        print(f"Question: {question}")

    rows = []
    for rank, (doc, score) in enumerate(results, start=1):
        rows.append(
            {
                "rank": rank,
                "source": doc.metadata.get("source", "unknown"),
                "score": round(float(score), 3),
                "preview": _clean_preview(doc.page_content, max_chars),
            }
        )

    df = pd.DataFrame(rows)
    styled = (
        df.style.background_gradient(
            cmap="RdYlGn_r",
            subset=["score"],
        )
        .set_properties(
            subset=["preview"],
            **{"text-align": "left", "white-space": "pre-wrap"},
        )
        .hide(axis="index")
    )
    return styled


def generate_answer(
    results,
    question: str,
    model: str = DEFAULT_CHAT_MODEL,
) -> str:
    """
    Feed the retrieved chunks to an LLM and return its answer as a string.
    results is the output of search(): a list of (Document, distance) tuples.
    The LLM is told to answer strictly from the provided context and to say
    it does not know if the context is missing the answer.
    """
    context = "\n\n---\n\n".join(doc.page_content for doc, _ in results)

    system = (
        "You answer questions using only the provided context. "
        "If the context does not contain the answer, say so plainly. "
        "Be concise. Include specific numbers and clauses when they exist."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"

    llm = ChatOpenAI(model=model, temperature=0)
    response = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
    return response.content.strip()


def pretty_print_chunks(chunks, max_chars: int = 150, n: int = 5):
    """
    Show the first n chunks from a splitter as a pandas table. Used in the
    chunking lesson to give students a concrete feel for what a chunk looks
    like before it becomes an embedding.
    """
    rows = []
    for i, chunk in enumerate(chunks[:n]):
        rows.append(
            {
                "index": i,
                "source": chunk.metadata.get("source", "unknown"),
                "length": len(chunk.page_content),
                "preview": _clean_preview(chunk.page_content, max_chars),
            }
        )

    df = pd.DataFrame(rows)
    styled = df.style.set_properties(
        subset=["preview"],
        **{"text-align": "left", "white-space": "pre-wrap"},
    ).hide(axis="index")
    return styled
