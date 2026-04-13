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


# ---------------------------------------------------------------------------
# Query translation helpers (Chapter 2)
#
# Each helper takes a user question and returns a transformed form of it or a
# list of retrieved documents. The notebook cells that use these are thin
# wrappers so the lesson is about the transformation, not the plumbing.
# ---------------------------------------------------------------------------


def _chat(model: str = DEFAULT_CHAT_MODEL, temperature: float = 0.0) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature)


def generate_query_variants(
    question: str,
    n: int = 3,
    context: str | None = None,
    model: str = DEFAULT_CHAT_MODEL,
) -> list[str]:
    """
    Ask the LLM to rewrite one vague question into n specific sub-questions.
    Returns a plain list of strings. Each variant targets a different angle
    of the original question so the downstream retrieval covers more ground.

    In production RAG, the rewriter always knows what product the questions
    are about. Pass a short context string describing your corpus so the
    rewrites reference your product by name instead of drifting into generic
    software questions.
    """
    system = (
        "You rewrite vague user questions into specific sub-questions that a "
        "vector database can answer. Each sub-question targets a different "
        "angle of the original. Use the product context when given, and "
        "reference the product by name in every sub-question. Return exactly "
        "the requested number of sub-questions, one per line, with no "
        "numbering, no bullets, no preamble. Do not repeat the original "
        "question."
    )
    context_block = f"Product context: {context}\n\n" if context else ""
    user = (
        f"{context_block}Original question: {question}\n\n"
        f"Rewrite this into {n} specific sub-questions, one per line."
    )
    response = _chat(model=model, temperature=0.2).invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    lines = [line.strip(" -\u2022\t") for line in response.content.splitlines()]
    variants = [line for line in lines if line]
    return variants[:n]


def multi_query_search(
    index,
    question: str,
    n: int = 3,
    k: int = 3,
    context: str | None = None,
    model: str = DEFAULT_CHAT_MODEL,
) -> tuple[list[str], list]:
    """
    Run multi-query retrieval. Generate n sub-questions with the LLM, then
    retrieve k results for each sub-question AND for the original question.
    Including the original query is the standard production pattern (see
    LangChain MultiQueryRetriever): it guarantees multi-query never performs
    worse than direct search. Results are deduped by (source, first 120 chars)
    and sorted by best (lowest) distance.

    Returns a tuple of (variants, merged_results) where variants is the list
    of LLM rewrites (excluding the original), so the notebook can print the
    rewrites and show the table in two clean cells.
    """
    variants = generate_query_variants(question, n=n, context=context, model=model)

    seen: dict = {}
    # Query with both the original question and each variant. Including the
    # original is what prevents the "multi-query drifted off-topic" failure
    # mode, where every variant is tangential and the core direct hit never
    # enters the result pool.
    for q in [question, *variants]:
        for doc, score in index.similarity_search_with_score(q, k=k):
            key = (doc.metadata.get("source", "unknown"), doc.page_content[:120])
            existing = seen.get(key)
            if existing is None or score < existing[1]:
                seen[key] = (doc, float(score))

    merged = sorted(seen.values(), key=lambda pair: pair[1])
    return variants, merged


def generate_hyde_doc(
    question: str,
    context: str | None = None,
    model: str = DEFAULT_CHAT_MODEL,
) -> str:
    """
    Ask the LLM to draft a short hypothetical answer to the question. The
    answer is intentionally plausible and formatted like internal docs, so
    its embedding lands near the real documents instead of near the user's
    vague question.
    """
    system = (
        "You write short hypothetical passages that sound like the internal "
        "documentation of a software product. Given a user question, write "
        "a two to three sentence passage that answers it as if you were "
        "quoting a real help article. Use concrete words. Reference the "
        "product by name when given. Do not hedge. Do not say you are "
        "speculating."
    )
    context_block = f"Product context: {context}\n\n" if context else ""
    user = f"{context_block}Question: {question}\n\nWrite the hypothetical passage."
    response = _chat(model=model, temperature=0.2).invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    return response.content.strip()


def hyde_search(
    index,
    question: str,
    k: int = 3,
    context: str | None = None,
    model: str = DEFAULT_CHAT_MODEL,
):
    """
    Run HyDE retrieval. Generate a hypothetical answer, then search the vector
    store using the hypothetical answer as the query. Returns a tuple of
    (hypothetical_text, results) where results is a list of (Document, distance).
    """
    hypo = generate_hyde_doc(question, context=context, model=model)
    return hypo, index.similarity_search_with_score(hypo, k=k)


def decompose_question(
    question: str,
    n: int = 3,
    model: str = DEFAULT_CHAT_MODEL,
) -> list[str]:
    """
    Break a complex multi-part question into independent sub-questions that
    can each be answered on their own. This is different from
    generate_query_variants: decomposition is for genuinely complex questions
    that contain multiple parts, not for vague questions that need rewording.
    """
    system = (
        "You decompose complex multi-part questions into independent "
        "sub-questions. Each sub-question should be answerable on its own, "
        "without reading the others. Return exactly the requested number of "
        "sub-questions, one per line, no numbering, no preamble."
    )
    user = (
        f"Complex question: {question}\n\n"
        f"Decompose into {n} independent sub-questions, one per line."
    )
    response = _chat(model=model, temperature=0.0).invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    lines = [line.strip(" -\u2022\t") for line in response.content.splitlines()]
    return [line for line in lines if line][:n]


def stepback_question(question: str, model: str = DEFAULT_CHAT_MODEL) -> str:
    """
    Generate a single broader "step-back" version of the question. The
    step-back version zooms out from the specific detail to the general topic,
    which retrieves foundational context the LLM can use to answer the
    original narrow question.
    """
    system = (
        "You are given a narrow specific question. Rewrite it as one broader "
        "question about the general topic or principle behind it. Return only "
        "the rewritten question, with no preamble."
    )
    user = f"Narrow question: {question}\n\nBroader version:"
    response = _chat(model=model, temperature=0.0).invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    return response.content.strip()


def show_queries(queries: list[str], title: str = "Generated queries"):
    """
    Pretty-print a list of query strings as a numbered pandas table. Used to
    show multi-query rewrites, decomposition output, or any other list of
    generated queries in a visual way that beats raw print().
    """
    df = pd.DataFrame(
        [{"#": i + 1, "query": q} for i, q in enumerate(queries)]
    )
    styled = (
        df.style.set_caption(title)
        .set_properties(
            subset=["query"],
            **{"text-align": "left", "white-space": "pre-wrap"},
        )
        .hide(axis="index")
    )
    return styled


# ---------------------------------------------------------------------------
# === Chapter 4 helpers ===
#
# Hybrid search combines BM25 keyword ranking with dense vector ranking and
# fuses the two result lists with Reciprocal Rank Fusion. Everything below is
# designed so the notebook cells stay short and the plumbing stays invisible.
# ---------------------------------------------------------------------------


def _bm25_tokenize(text: str) -> list[str]:
    """
    Lowercase the text and split on anything that is not a letter, digit, or
    hyphen. Hyphens are kept inside tokens so identifiers like E-4012 and
    SKU-7829 survive as single tokens. This matches how product teams
    usually index error codes and SKUs for keyword search.
    """
    import re

    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def build_bm25_index(chunks: list) -> dict:
    """
    Build a BM25 keyword index over the given chunks. Returns a dict with
    the fitted BM25Okapi object, the original chunks, and the tokenizer used
    to build the index. The notebook passes this dict to bm25_search().

    BM25 scores are higher for better matches, which is the opposite of the
    FAISS L2 distance convention. The notebook keeps both conventions
    visible so students see the difference, and hybrid_search normalizes
    both into rank space before fusing.
    """
    from rank_bm25 import BM25Okapi

    tokenized = [_bm25_tokenize(c.page_content) for c in chunks]
    bm25 = BM25Okapi(tokenized)
    print(f"Built BM25 index over {len(chunks)} chunks")
    return {
        "bm25": bm25,
        "chunks": chunks,
        "tokenize": _bm25_tokenize,
    }


def bm25_search(bm25_index: dict, question: str, k: int = 5) -> list:
    """
    Run a BM25 query and return the top k chunks as a list of
    (Document, score) pairs. Higher score means a better keyword match.
    The return shape mirrors search() so the notebook can feed BM25 results
    through show_results() and generate_answer() the same way.
    """
    bm25 = bm25_index["bm25"]
    chunks = bm25_index["chunks"]
    tokenize = bm25_index["tokenize"]

    scores = bm25.get_scores(tokenize(question))
    ranked_indices = sorted(range(len(chunks)), key=lambda i: -scores[i])[:k]
    return [(chunks[i], float(scores[i])) for i in ranked_indices]


def reciprocal_rank_fusion(
    ranked_lists: list[list],
    k: int = 60,
    weights: list[float] | None = None,
) -> list:
    """
    Merge several ranked result lists into one list using Reciprocal Rank
    Fusion. Each input list is a list of (Document, score) pairs already
    sorted from best to worst. The RRF score for a document that appears at
    rank r in a list is 1 / (k + r), summed across every list it appears in.
    Higher total means better.

    Documents are deduplicated across lists by (source, first 120 characters
    of content). If weights is passed, each list's contribution is multiplied
    by its weight, which is how alpha tuning in hybrid_search is implemented.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    if len(weights) != len(ranked_lists):
        raise ValueError("weights must have the same length as ranked_lists")

    fused: dict = {}
    for ranked, weight in zip(ranked_lists, weights):
        for rank, (doc, _score) in enumerate(ranked, start=1):
            key = (doc.metadata.get("source", "unknown"), doc.page_content[:120])
            contribution = weight * (1.0 / (k + rank))
            if key in fused:
                existing_doc, existing_score = fused[key]
                fused[key] = (existing_doc, existing_score + contribution)
            else:
                fused[key] = (doc, contribution)

    return sorted(fused.values(), key=lambda pair: -pair[1])


def hybrid_search(
    vector_index,
    bm25_index: dict,
    question: str,
    k: int = 5,
    alpha: float = 0.5,
    rrf_k: int = 60,
    pool_size: int = 10,
) -> list:
    """
    Run the question against both the vector index and the BM25 index,
    retrieve pool_size candidates from each path, and fuse the two ranked
    lists with Reciprocal Rank Fusion. Alpha sets the mix: alpha=1.0 is
    pure vector, alpha=0.0 is pure BM25, alpha=0.5 is an even blend. The
    pool_size parameter controls how many candidates each path contributes
    before fusion, and k controls how many fused results come out the other
    side.

    Returns a list of (Document, rrf_score) pairs sorted by fused score,
    descending. Higher rrf_score means the document ranked well across both
    paths.
    """
    vector_ranked = vector_index.similarity_search_with_score(question, k=pool_size)
    bm25_ranked = bm25_search(bm25_index, question, k=pool_size)

    fused = reciprocal_rank_fusion(
        [vector_ranked, bm25_ranked],
        k=rrf_k,
        weights=[alpha, 1.0 - alpha],
    )
    return fused[:k]


def show_bm25_results(results, question: str | None = None, max_chars: int = 200):
    """
    Pretty-print a BM25 result list as a pandas table. BM25 scores are
    rewards, not distances, so the gradient is flipped compared to
    show_results: green is the highest score (best match) and red is the
    lowest. Shape matches show_results so the notebook can place the two
    tables side by side.
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
            cmap="RdYlGn",
            subset=["score"],
        )
        .set_properties(
            subset=["preview"],
            **{"text-align": "left", "white-space": "pre-wrap"},
        )
        .hide(axis="index")
    )
    return styled


def show_hybrid_results(results, question: str | None = None, max_chars: int = 200):
    """
    Pretty-print a hybrid (RRF-fused) result list. RRF scores are small
    positive numbers where higher is better, so the gradient goes green at
    the top. The score column is formatted to four decimal places because
    RRF values are typically in the 0.01 to 0.05 range.
    """
    if question:
        print(f"Question: {question}")

    rows = []
    for rank, (doc, score) in enumerate(results, start=1):
        rows.append(
            {
                "rank": rank,
                "source": doc.metadata.get("source", "unknown"),
                "rrf_score": round(float(score), 4),
                "preview": _clean_preview(doc.page_content, max_chars),
            }
        )

    df = pd.DataFrame(rows)
    styled = (
        df.style.background_gradient(
            cmap="RdYlGn",
            subset=["rrf_score"],
        )
        .set_properties(
            subset=["preview"],
            **{"text-align": "left", "white-space": "pre-wrap"},
        )
        .hide(axis="index")
    )
    return styled


# === end Chapter 4 helpers ===
