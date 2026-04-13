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


# === Chapter 3 helpers ===
#
# Routing sends each question to the right data source. Chapter 3 uses three
# mock sources for SkillAgents AI:
#   1. The FAISS vector store from earlier chapters (unstructured docs)
#   2. A pandas DataFrame of quarterly revenue by segment (SQL-ish)
#   3. A dict-based org chart (graph-ish)
#
# The helpers below hide all the loading and dispatch plumbing so the lesson
# cells can stay short. Students see the router decide, then see the answer.


def load_revenue_table(path: str = "data/skillagents/ch3_revenue.csv"):
    """
    Load the Chapter 3 SkillAgents revenue table as a pandas DataFrame.
    Columns: quarter, segment, revenue_usd, paying_customers, new_signups.
    """
    return pd.read_csv(path)


def load_org_chart(
    path: str = "data/skillagents/ch3_org_chart.py",
):
    """
    Load the Chapter 3 org chart module and return a dict with three keys:
    graph (the adjacency dict), direct_reports (function), find_person
    (function). Students do not need to worry about importing by file path.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_ch3_org_chart", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"Could not load org chart module at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {
        "graph": module.ORG_CHART,
        "direct_reports": module.direct_reports,
        "find_person": module.find_person,
    }


# Default routes for Chapter 3. Students can override these or extend them.
DEFAULT_CH3_ROUTES = {
    "vector_store": (
        "Questions about product documentation, policies, pricing plans, "
        "billing mechanics, error codes, refund terms, or anything answered "
        "by prose in a help article or PDF."
    ),
    "revenue_table": (
        "Questions about company revenue, paying customers, signups, or any "
        "numerical metric broken down by quarter or by segment "
        "(Student, Pro, Enterprise). These need a pandas lookup on the "
        "quarterly revenue table."
    ),
    "org_chart": (
        "Questions about who reports to whom, team structure, direct reports, "
        "or management chains. These need a graph traversal on the org chart."
    ),
}


def classify_route_llm(
    question: str,
    routes: dict = None,
    context: str | None = None,
    model: str = DEFAULT_CHAT_MODEL,
) -> str:
    """
    Ask an LLM to pick the best route for a single question. routes is a dict
    mapping route name to a one-sentence description of what that route is
    good for. Returns the winning route name as a string. The LLM is told to
    pick exactly one route and return only the route name with no punctuation.

    In production RAG, the classifier always knows what product the questions
    are about. Pass a short context string describing your corpus and your
    data sources so the classifier does not drift.
    """
    if routes is None:
        routes = DEFAULT_CH3_ROUTES

    route_block = "\n".join(
        f"- {name}: {description}" for name, description in routes.items()
    )
    valid_names = ", ".join(routes.keys())

    system = (
        "You are a routing classifier for a retrieval system. You read one "
        "user question and pick exactly one route name from the provided "
        "list. Return only the route name. No quotes. No punctuation. No "
        "explanation. If the question could fit multiple routes, pick the "
        "one that answers the most important part of the question."
    )
    context_block = f"Product context: {context}\n\n" if context else ""
    user = (
        f"{context_block}Available routes:\n{route_block}\n\n"
        f"Valid route names: {valid_names}\n\n"
        f"Question: {question}\n\nRoute name:"
    )

    response = _chat(model=model, temperature=0.0).invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    raw = response.content.strip().strip(".,:;\"' \n").lower().replace(" ", "_")
    # Accept either the exact route name or a close match.
    for name in routes.keys():
        if name.lower() == raw:
            return name
    for name in routes.keys():
        if name.lower() in raw or raw in name.lower():
            return name
    # Fall back to the first route so the pipeline does not crash on a drift.
    return next(iter(routes.keys()))


def query_revenue_table(
    question: str,
    revenue_df,
    model: str = DEFAULT_CHAT_MODEL,
) -> dict:
    """
    Ask an LLM to turn one question into a pandas filter and aggregation
    against the revenue DataFrame. The LLM returns a small JSON plan. This
    helper executes the plan and returns a dict with the plan, the filtered
    rows, and a one-line natural language summary. Students see the plan AND
    the rows, so they understand what SQL-ish routing actually does.
    """
    import json

    columns = list(revenue_df.columns)
    quarters = sorted(revenue_df["quarter"].unique().tolist())
    segments = sorted(revenue_df["segment"].unique().tolist())

    system = (
        "You translate one user question into a JSON plan that filters a "
        "pandas DataFrame of quarterly company revenue. Return only JSON. "
        "The JSON must have these keys: quarters (list of strings, empty "
        "means all), segments (list of strings, empty means all), metric "
        "(one of revenue_usd, paying_customers, new_signups), aggregation "
        "(one of sum, mean, max, min, none). Use none when the user wants "
        "the raw row or rows without any aggregation."
    )
    user = (
        f"DataFrame columns: {columns}\n"
        f"Valid quarters: {quarters}\n"
        f"Valid segments: {segments}\n\n"
        f"Question: {question}\n\nJSON plan:"
    )

    response = _chat(model=model, temperature=0.0).invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    raw = response.content.strip()
    # Strip code fences if the LLM wrapped the JSON.
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:]
        raw = raw.strip()
    plan = json.loads(raw)

    filtered = revenue_df
    if plan.get("quarters"):
        filtered = filtered[filtered["quarter"].isin(plan["quarters"])]
    if plan.get("segments"):
        filtered = filtered[filtered["segment"].isin(plan["segments"])]

    metric = plan.get("metric", "revenue_usd")
    agg = plan.get("aggregation", "none")

    if agg == "sum":
        value = float(filtered[metric].sum())
        summary = (
            f"Total {metric} across the filtered rows is {value:,.0f}."
        )
    elif agg == "mean":
        value = float(filtered[metric].mean())
        summary = f"Average {metric} is {value:,.0f}."
    elif agg == "max":
        value = float(filtered[metric].max())
        summary = f"Max {metric} is {value:,.0f}."
    elif agg == "min":
        value = float(filtered[metric].min())
        summary = f"Min {metric} is {value:,.0f}."
    else:
        value = None
        summary = f"Returned {len(filtered)} matching row(s) for {metric}."

    return {
        "plan": plan,
        "rows": filtered.reset_index(drop=True),
        "value": value,
        "summary": summary,
    }


def query_org_chart(
    question: str,
    org_pack: dict,
    model: str = DEFAULT_CHAT_MODEL,
) -> dict:
    """
    Use an LLM to pick which person the question is asking about, then look
    up their direct reports in the org chart. Returns a dict with the matched
    person, the list of direct reports, and a one-line summary.
    """
    people = list(org_pack["graph"].keys())
    people_block = "\n".join(f"- {p}" for p in people)

    system = (
        "You read one question about an org chart and pick the single "
        "manager whose direct reports the question is asking about. If the "
        "user asks 'who reports to X', return X. If the user asks 'who does "
        "X manage', return X. Return only the exact label from the provided "
        "list. No quotes. No punctuation. No explanation."
    )
    user = (
        f"Managers in the org chart:\n{people_block}\n\n"
        f"Question: {question}\n\nManager label:"
    )
    response = _chat(model=model, temperature=0.0).invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    raw = response.content.strip().strip("\"'.,")
    # Try exact match first, then a fuzzy find.
    if raw in org_pack["graph"]:
        person = raw
    else:
        person = org_pack["find_person"](raw) or org_pack["find_person"](
            question
        ) or people[0]

    reports = org_pack["direct_reports"](person)
    if reports:
        summary = (
            f"{person} has {len(reports)} direct report(s): "
            + ", ".join(reports)
            + "."
        )
    else:
        summary = f"{person} has no direct reports recorded in the graph."

    return {
        "person": person,
        "reports": reports,
        "summary": summary,
    }


def run_route(
    route: str,
    question: str,
    context_pack: dict,
    k: int = 3,
) -> dict:
    """
    Dispatch one question to the correct data source based on the route name.
    context_pack carries the three sources in one place:
      { "index": faiss_index, "revenue_df": df, "org_pack": org_pack }
    Returns a dict with keys: route, result_text, detail. result_text is a
    short string ready to feed into the LLM. detail holds the raw object for
    rendering (list of chunks, filtered DataFrame, or direct reports list).
    """
    if route == "vector_store":
        results = context_pack["index"].similarity_search_with_score(
            question, k=k
        )
        joined = "\n\n---\n\n".join(doc.page_content for doc, _ in results)
        return {
            "route": route,
            "result_text": joined,
            "detail": results,
        }
    if route == "revenue_table":
        out = query_revenue_table(question, context_pack["revenue_df"])
        rows_text = out["rows"].to_string(index=False)
        body = f"Plan: {out['plan']}\nRows:\n{rows_text}\n{out['summary']}"
        return {
            "route": route,
            "result_text": body,
            "detail": out,
        }
    if route == "org_chart":
        out = query_org_chart(question, context_pack["org_pack"])
        body = out["summary"]
        return {
            "route": route,
            "result_text": body,
            "detail": out,
        }
    raise ValueError(f"Unknown route: {route}")


def answer_from_route(
    question: str,
    route_output: dict,
    model: str = DEFAULT_CHAT_MODEL,
) -> str:
    """
    Feed the result_text from run_route into the LLM and return a final
    answer string. This is the same pattern as generate_answer but works for
    any source, not only vector retrieval.
    """
    system = (
        "You answer questions using only the provided context. If the "
        "context does not contain the answer, say so plainly. Be concise. "
        "Include specific numbers and names when they exist."
    )
    user = (
        f"Source: {route_output['route']}\n\n"
        f"Context:\n{route_output['result_text']}\n\n"
        f"Question: {question}"
    )
    llm = ChatOpenAI(model=model, temperature=0)
    response = llm.invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    return response.content.strip()


def route_and_answer(
    question: str,
    context_pack: dict,
    routes: dict = None,
    context: str | None = None,
    model: str = DEFAULT_CHAT_MODEL,
) -> dict:
    """
    End-to-end single-source routing. Classify the question with the LLM
    router, run the winning route, generate the final answer. Returns a dict
    with route, route_output, and answer. Students can print each field to
    see what happened at every step.
    """
    if routes is None:
        routes = DEFAULT_CH3_ROUTES
    route = classify_route_llm(
        question, routes=routes, context=context, model=model
    )
    route_output = run_route(route, question, context_pack)
    answer = answer_from_route(question, route_output, model=model)
    return {
        "route": route,
        "route_output": route_output,
        "answer": answer,
    }


def build_semantic_router(route_examples: dict):
    """
    Pre-compute embeddings for a semantic router. route_examples is a dict
    mapping route name to a list of example phrases. Returns a dict with the
    route names, the embedding matrix, and the row-to-route index. Call
    semantic_route with this bundle and a question to get a ranking.
    """
    import numpy as np

    embeddings = OpenAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL)
    labels: list = []
    phrases: list = []
    for name, examples in route_examples.items():
        for ex in examples:
            labels.append(name)
            phrases.append(ex)

    vectors = embeddings.embed_documents(phrases)
    matrix = np.array(vectors, dtype="float32")
    # Normalize so dot product equals cosine similarity.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.clip(norms, 1e-9, None)

    return {
        "labels": labels,
        "phrases": phrases,
        "matrix": matrix,
        "routes": list(route_examples.keys()),
        "_embeddings": embeddings,
    }


def semantic_route(
    bundle: dict,
    question: str,
) -> dict:
    """
    Score one question against a pre-computed semantic router. Returns a
    dict with the winning route name, a per-route top score, and the raw
    per-example similarity table. Cosine similarity. Higher is better.
    """
    import numpy as np

    q_vec = np.array(
        bundle["_embeddings"].embed_query(question), dtype="float32"
    )
    q_vec = q_vec / max(np.linalg.norm(q_vec), 1e-9)

    sims = bundle["matrix"] @ q_vec  # cosine similarity per example

    # Collapse per-example scores to per-route top score.
    top_per_route: dict = {r: -1.0 for r in bundle["routes"]}
    for label, sim in zip(bundle["labels"], sims):
        if sim > top_per_route[label]:
            top_per_route[label] = float(sim)

    winner = max(top_per_route, key=top_per_route.get)

    table = pd.DataFrame(
        {
            "route": bundle["labels"],
            "example": bundle["phrases"],
            "similarity": [round(float(s), 3) for s in sims],
        }
    ).sort_values("similarity", ascending=False).reset_index(drop=True)

    return {
        "winner": winner,
        "scores": top_per_route,
        "table": table,
    }


def show_route_scores(scores: dict):
    """
    Render a per-route score dict as a styled pandas table sorted highest
    first. Used after classify_route_llm for a quick visual, or after
    semantic_route for the collapsed per-route view.
    """
    rows = [
        {"route": name, "score": round(float(score), 3)}
        for name, score in scores.items()
    ]
    df = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(
        drop=True
    )
    return (
        df.style.background_gradient(cmap="RdYlGn", subset=["score"])
        .hide(axis="index")
    )

# === end Chapter 3 helpers ===
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

# === Chapter 5 helpers ===
#
# These helpers wrap Cohere Rerank so the Chapter 5 notebook can re-sort an
# existing list of (Document, distance) results without introducing a new
# data type. Everything stays as lists of (Document, score) tuples that the
# rest of the shared helpers already know how to render.


def rerank_with_cohere(
    question: str,
    results: list,
    top_n: int = 3,
    model: str = "rerank-english-v3.0",
) -> list:
    """
    Re-sort a list of (Document, distance) results with Cohere Rerank.

    Takes the output of `search()` and sends the question plus every document
    text to Cohere's hosted cross-encoder. Cohere returns a relevance score
    in the zero to one range for each document, where one is a perfect
    match. This function returns a new list of (Document, relevance_score)
    tuples sorted from highest to lowest relevance, keeping only the top_n.

    The output shape matches `search()` so you can pass the result straight
    into `show_results()` or `generate_answer()` without any adapter code.
    Note that the score column now holds relevance scores from Cohere, not
    L2 distances from FAISS. Higher is better here, the opposite of search().
    """
    # Local import so the rest of the notebooks never pay the cohere import
    # cost when they are not using rerank.
    import cohere

    client = cohere.Client(api_key=os.environ.get("COHERE_API_KEY"))

    documents = [doc.page_content for doc, _ in results]
    response = client.rerank(
        query=question,
        documents=documents,
        top_n=top_n,
        model=model,
    )

    reranked: list = []
    for item in response.results:
        original_doc = results[item.index][0]
        reranked.append((original_doc, float(item.relevance_score)))
    return reranked


def show_rerank_results(results, question: str | None = None, max_chars: int = 200):
    """
    Like show_results() but colored for Cohere relevance scores instead of
    L2 distances. Higher relevance means a better match, so the gradient
    runs green-high, red-low, the opposite of show_results(). Used to render
    the output of rerank_with_cohere().
    """
    if question:
        print(f"Question: {question}")

    rows = []
    for rank, (doc, score) in enumerate(results, start=1):
        rows.append(
            {
                "rank": rank,
                "source": doc.metadata.get("source", "unknown"),
                "relevance": round(float(score), 4),
                "preview": _clean_preview(doc.page_content, max_chars),
            }
        )

    df = pd.DataFrame(rows)
    styled = (
        df.style.background_gradient(
            cmap="RdYlGn",
            subset=["relevance"],
        )
        .set_properties(
            subset=["preview"],
            **{"text-align": "left", "white-space": "pre-wrap"},
        )
        .hide(axis="index")
    )
    return styled


# === end Chapter 5 helpers ===

# === Chapter 6 helpers ===
#
# Evaluation helpers built on RAGAS. The functions below hide the RAGAS
# EvaluationDataset, judge LLM, and embeddings wiring so the notebook cells
# stay three to ten lines long. ragas is imported lazily so earlier chapter
# notebooks do not pay the import cost.


_RAGAS_METRIC_COLUMNS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
]


def show_test_set(questions: list[dict]):
    """
    Render the Chapter 6 test set as a styled pandas table. Each row has the
    question id, the question text, the expected sources, and a short
    preview of the ground truth answer so students can eyeball the whole set
    at once.
    """
    rows = []
    for q in questions:
        rows.append(
            {
                "id": q.get("id", ""),
                "question": q.get("question", ""),
                "expected_sources": ", ".join(q.get("expected_sources", [])),
                "ground_truth": _clean_preview(q.get("ground_truth", ""), 160),
            }
        )
    df = pd.DataFrame(rows)
    styled = (
        df.style.set_properties(
            subset=["question", "ground_truth"],
            **{"text-align": "left", "white-space": "pre-wrap"},
        )
        .hide(axis="index")
    )
    return styled


def _style_ragas_df(df):
    """Apply a red to green gradient across the four RAGAS metric columns."""
    metric_cols = [c for c in _RAGAS_METRIC_COLUMNS if c in df.columns]
    # vmin=0 vmax=1 forces a stable color scale across runs, so students see
    # the same green for 0.9 every time regardless of the per run min and max.
    styled = df.style.background_gradient(
        cmap="RdYlGn",
        subset=metric_cols,
        vmin=0.0,
        vmax=1.0,
    )
    preview_cols = [c for c in ["user_input", "response", "reference"] if c in df.columns]
    if preview_cols:
        styled = styled.set_properties(
            subset=preview_cols,
            **{"text-align": "left", "white-space": "pre-wrap"},
        )
    styled = styled.format({c: "{:.3f}" for c in metric_cols}).hide(axis="index")
    return styled


def _build_ragas_samples(rows):
    """Convert a list of dicts with question, contexts, answer, and ground_truth
    into a ragas EvaluationDataset of SingleTurnSample objects."""
    from ragas import EvaluationDataset
    from ragas.dataset_schema import SingleTurnSample

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            retrieved_contexts=list(r["contexts"]),
            response=r["answer"],
            reference=r["ground_truth"],
        )
        for r in rows
    ]
    return EvaluationDataset(samples=samples)


def _ragas_metrics():
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    return [faithfulness, answer_relevancy, context_precision, context_recall]


def _wrapped_judge(model: str):
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    judge_llm = LangchainLLMWrapper(ChatOpenAI(model=model, temperature=0))
    judge_embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model=DEFAULT_EMBEDDING_MODEL)
    )
    return judge_llm, judge_embeddings


class _DisableLangsmith:
    """
    Context manager that turns LangSmith tracing off for the duration of a
    RAGAS call. RAGAS makes dozens of LLM calls per row internally, and the
    LangChain tracer prints a lot of warnings while they run. Disabling the
    tracer on entry and restoring the prior value on exit keeps the notebook
    readable without losing the trace on everything else.
    """

    def __enter__(self):
        self._prev_tracing = os.environ.get("LANGCHAIN_TRACING_V2")
        self._prev_endpoint = os.environ.get("LANGCHAIN_ENDPOINT")
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        os.environ["LANGCHAIN_ENDPOINT"] = ""
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._prev_tracing is None:
            os.environ.pop("LANGCHAIN_TRACING_V2", None)
        else:
            os.environ["LANGCHAIN_TRACING_V2"] = self._prev_tracing
        if self._prev_endpoint is None:
            os.environ.pop("LANGCHAIN_ENDPOINT", None)
        else:
            os.environ["LANGCHAIN_ENDPOINT"] = self._prev_endpoint


def run_ragas_eval(
    test_set: list[dict],
    pipeline_fn,
    judge_model: str = DEFAULT_CHAT_MODEL,
):
    """
    Run a full RAGAS evaluation over a test set.

    Parameters
    ----------
    test_set : list of dicts
        Each entry must have keys "question" and "ground_truth". The
        "expected_sources" key is optional and is ignored by RAGAS.
    pipeline_fn : callable
        Function that takes a question string and returns a tuple of
        (contexts_list, answer_string). This is the RAG pipeline under test.
    judge_model : str
        Name of the chat model that RAGAS should use as its judge. Defaults
        to gpt-4o-mini because it is cheap and stable.

    Returns
    -------
    (styled_df, raw_df) : tuple
        styled_df is a pandas Styler ready for display in a notebook, with
        a red to green gradient across the metric columns. raw_df is the
        plain numeric DataFrame for further computation.
    """
    from ragas import evaluate

    rows = []
    for entry in test_set:
        question = entry["question"]
        ground_truth = entry["ground_truth"]
        contexts, answer = pipeline_fn(question)
        rows.append(
            {
                "question": question,
                "ground_truth": ground_truth,
                "contexts": contexts,
                "answer": answer,
            }
        )

    dataset = _build_ragas_samples(rows)
    judge_llm, judge_embeddings = _wrapped_judge(judge_model)

    with _DisableLangsmith():
        result = evaluate(
            dataset=dataset,
            metrics=_ragas_metrics(),
            llm=judge_llm,
            embeddings=judge_embeddings,
            show_progress=False,
        )

    df = result.to_pandas()
    return _style_ragas_df(df), df


def score_single_sample(
    question: str,
    contexts: list[str],
    answer: str,
    ground_truth: str,
    judge_model: str = DEFAULT_CHAT_MODEL,
):
    """
    Score a single (question, contexts, answer, ground_truth) tuple with
    RAGAS and return a styled DataFrame plus the raw DataFrame. Used for
    the broken answer demo where the input is a single hand crafted row.
    """
    from ragas import evaluate

    rows = [
        {
            "question": question,
            "ground_truth": ground_truth,
            "contexts": contexts,
            "answer": answer,
        }
    ]
    dataset = _build_ragas_samples(rows)
    judge_llm, judge_embeddings = _wrapped_judge(judge_model)

    with _DisableLangsmith():
        result = evaluate(
            dataset=dataset,
            metrics=_ragas_metrics(),
            llm=judge_llm,
            embeddings=judge_embeddings,
            show_progress=False,
        )
    df = result.to_pandas()
    return _style_ragas_df(df), df


def show_score_averages(df, label: str = "run"):
    """
    Compute mean scores across a RAGAS result DataFrame and render them as a
    one row styled table with the same red to green gradient used on the
    full scorecard. Missing metric columns are skipped silently.
    """
    metric_cols = [c for c in _RAGAS_METRIC_COLUMNS if c in df.columns]
    averages = {c: round(float(df[c].mean()), 3) for c in metric_cols}
    averages = {"run": label, **averages}
    avg_df = pd.DataFrame([averages])
    styled = (
        avg_df.style.background_gradient(
            cmap="RdYlGn",
            subset=metric_cols,
            vmin=0.0,
            vmax=1.0,
        )
        .format({c: "{:.3f}" for c in metric_cols})
        .hide(axis="index")
    )
    return styled


# === end Chapter 6 helpers ===

# === Chapter 7 helpers ===
# Self-RAG (agentic RAG) helpers. Build a LangGraph state machine that
# retrieves, generates, grades the answer for groundedness, and loops back on
# failure. Max retries is bounded so the graph always terminates.
#
# Design notes the notebook depends on:
#
#   1. The grader scores the answer on a 1-5 integer scale. A score at or
#      above the threshold passes. Below the threshold fails and triggers
#      a retry.
#
#   2. The grader is explicitly told that refund_policy.pdf is the single
#      authoritative source for refund questions in this corpus. If the
#      answer conflicts with that source, the grader must fail it even if
#      the rest of the context appears to support the answer. This is the
#      conflict-aware check that catches the outdated blog post.
#
#   3. On retry, the rewriter prepends a reference to the official refund
#      policy document to the query. That biases the vector search toward
#      the authoritative PDF chunks and away from the outdated blog post.
#
#   4. run_self_rag prints a node-by-node trace as the graph runs so the
#      notebook reader can see every step of the loop.


def _authoritative_sources_for_question(question: str) -> list[str]:
    """
    Return the filenames this corpus treats as ground truth for the given
    question. For the SkillAgents corpus, refund questions are grounded in
    refund_policy.pdf. Other topics have no special authoritative source.
    """
    q = question.lower()
    refund_words = ["refund", "cancel", "money back", "guarantee"]
    if any(word in q for word in refund_words):
        return ["refund_policy.pdf"]
    return []


def grade_answer(
    question: str,
    answer: str,
    context_docs: list,
    threshold: int = 4,
    model: str = DEFAULT_CHAT_MODEL,
) -> dict:
    """
    Ask an LLM to grade whether the answer is grounded in the retrieved
    context and consistent with the authoritative source for the question.
    Returns a dict with keys score (1 to 5), passed (bool), and reason.

    The grader is shown three things: the question, the candidate answer,
    and the retrieved passages. It is instructed that refund_policy.pdf is
    the single source of truth for refund topics and that any answer which
    conflicts with that source must receive a low score, even if other
    retrieved passages appear to support it. This is how the loop catches
    the stale blog post claim of a 30 day full refund.
    """
    authoritative = _authoritative_sources_for_question(question)

    context_block_parts = []
    for i, doc in enumerate(context_docs, start=1):
        src = doc.metadata.get("source", "unknown")
        mark = " [AUTHORITATIVE]" if src in authoritative else ""
        context_block_parts.append(
            f"Passage {i} from {src}{mark}:\n{doc.page_content}"
        )
    context_block = "\n\n---\n\n".join(context_block_parts)

    authoritative_note = ""
    if authoritative:
        authoritative_note = (
            "For this question, the single source of truth is "
            f"{', '.join(authoritative)}. If the candidate answer conflicts "
            "with that source, score it 1 or 2 even if other passages seem "
            "to support the answer. Outdated blog posts, announcements, and "
            "marketing copy are not authoritative.\n\n"
        )

    system = (
        "You are a strict grader for a retrieval augmented generation "
        "pipeline. You score whether a candidate answer is grounded in the "
        "retrieved passages and consistent with the authoritative source. "
        "You respond with a single integer from 1 to 5 followed by a short "
        "reason on the next line. "
        "5 means fully grounded and consistent with the authoritative source. "
        "4 means grounded with minor gaps. "
        "3 means partially grounded. "
        "2 means the answer conflicts with the authoritative source on a "
        "material point. "
        "1 means the answer is unsupported or contradicted by the "
        "authoritative source."
    )
    user = (
        f"{authoritative_note}"
        f"Question: {question}\n\n"
        f"Candidate answer:\n{answer}\n\n"
        f"Retrieved passages:\n{context_block}\n\n"
        "Score the answer on a 1 to 5 scale. First line is the integer. "
        "Second line is a one sentence reason."
    )

    llm = _chat(model=model, temperature=0.0)
    response = llm.invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    ).content.strip()

    lines = [line.strip() for line in response.splitlines() if line.strip()]
    score = 1
    reason = response
    if lines:
        first = lines[0].split()[0] if lines[0] else "1"
        try:
            score = int(first)
        except ValueError:
            score = 1
        if len(lines) > 1:
            reason = lines[1]
        else:
            reason = lines[0]

    score = max(1, min(5, score))
    return {
        "score": score,
        "passed": score >= threshold,
        "reason": reason,
    }


def rewrite_for_authoritative_source(
    question: str,
    prior_answer: str,
    reason: str,
    model: str = DEFAULT_CHAT_MODEL,
) -> str:
    """
    Rewrite the original question so the next retrieval prefers the
    authoritative source in the corpus. For refund questions, the rewrite
    explicitly names refund_policy.pdf so the embedding model lands on
    official policy chunks instead of the outdated marketing blog post.
    """
    authoritative = _authoritative_sources_for_question(question)
    source_hint = (
        f"the official document named {authoritative[0]}"
        if authoritative
        else "the official authoritative policy document"
    )

    system = (
        "You rewrite a user question so that a vector search retrieves the "
        "official authoritative document instead of outdated marketing or "
        "blog content. Keep the user intent intact. Add explicit references "
        "to the official policy source. Return only the rewritten question."
    )
    user = (
        f"Original question: {question}\n\n"
        f"The previous answer was flagged as ungrounded. Reason: {reason}\n\n"
        f"Rewrite the question so that retrieval prefers {source_hint}. "
        "Include the phrase 'according to the official refund policy document' "
        "when the question is about refunds, cancellations, or money back."
    )
    response = _chat(model=model, temperature=0.0).invoke(
        [SystemMessage(content=system), HumanMessage(content=user)]
    )
    return response.content.strip().strip('"').strip("'")


def _authoritative_biased_search(
    index,
    question: str,
    k: int = 3,
    authoritative_sources: list[str] | None = None,
):
    """
    Retrieve k results. If any authoritative sources are named for the
    question, fetch a much wider pool (k * 10) and filter down to only the
    authoritative chunks, returning the top k by distance from those. This
    guarantees the loop sees a focused authoritative context on retry, with
    no stale documents or decoy FAQs mixed in.

    This is a stricter bias than a soft re-ranking. The lesson of self-RAG
    is that you can route retrieval away from sources that the grader flagged
    as conflicting. Filtering is the cleanest expression of that idea.

    If the wide pool somehow contains zero authoritative chunks (for example
    because the question does not actually match the authoritative source
    well), fall back to the wide pool so the pipeline still returns something.
    """
    if not authoritative_sources:
        return index.similarity_search_with_score(question, k=k)

    wide = index.similarity_search_with_score(question, k=max(k * 10, 20))
    auth = [(d, s) for d, s in wide if d.metadata.get("source") in authoritative_sources]
    if not auth:
        return wide[:k]
    return auth[:k]


def build_self_rag_graph(
    index,
    max_retries: int = 2,
    grader_threshold: int = 4,
    k: int = 3,
):
    """
    Build a compiled LangGraph self-RAG state machine over the given FAISS
    index. The graph has four nodes: retrieve, generate, grade, and rewrite.
    The edge from grade is conditional: pass routes to END, fail routes to
    rewrite which then loops back to retrieve. The loop is bounded by
    max_retries so the graph always terminates.

    Returns a compiled graph that run_self_rag can stream.
    """
    from typing import TypedDict
    from langgraph.graph import StateGraph, END

    class SelfRAGState(TypedDict, total=False):
        question: str
        current_query: str
        retries: int
        retrieved: list
        answer: str
        score: int
        passed: bool
        reason: str
        trace: list

    def retrieve_node(state: SelfRAGState) -> SelfRAGState:
        q = state["current_query"]
        # On a retry the query has been rewritten to prefer the official
        # source, so bias the retriever toward the authoritative document.
        on_retry = state.get("retries", 0) > 0
        sources = (
            _authoritative_sources_for_question(state["question"]) if on_retry else []
        )
        results = _authoritative_biased_search(
            index,
            q,
            k=k,
            authoritative_sources=sources,
        )
        docs = [d for d, _ in results]
        trace = state.get("trace", []) + [
            {
                "node": "retrieve",
                "attempt": state.get("retries", 0) + 1,
                "query": q,
                "top_sources": [d.metadata.get("source", "unknown") for d in docs],
            }
        ]
        return {"retrieved": docs, "trace": trace}

    def generate_node(state: SelfRAGState) -> SelfRAGState:
        docs = state["retrieved"]
        context = "\n\n---\n\n".join(d.page_content for d in docs)
        system = (
            "You answer questions using only the provided context. "
            "If the context does not contain the answer, say so plainly. "
            "Be concise. Include specific numbers and clauses when they exist."
        )
        user = f"Context:\n{context}\n\nQuestion: {state['question']}"
        llm = _chat(temperature=0.0)
        answer = llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=user)]
        ).content.strip()
        trace = state.get("trace", []) + [
            {
                "node": "generate",
                "attempt": state.get("retries", 0) + 1,
                "answer": answer,
            }
        ]
        return {"answer": answer, "trace": trace}

    def grade_node(state: SelfRAGState) -> SelfRAGState:
        verdict = grade_answer(
            state["question"],
            state["answer"],
            state["retrieved"],
            threshold=grader_threshold,
        )
        trace = state.get("trace", []) + [
            {
                "node": "grade",
                "attempt": state.get("retries", 0) + 1,
                "score": verdict["score"],
                "passed": verdict["passed"],
                "reason": verdict["reason"],
            }
        ]
        return {
            "score": verdict["score"],
            "passed": verdict["passed"],
            "reason": verdict["reason"],
            "trace": trace,
        }

    def rewrite_node(state: SelfRAGState) -> SelfRAGState:
        new_query = rewrite_for_authoritative_source(
            state["question"],
            state.get("answer", ""),
            state.get("reason", ""),
        )
        trace = state.get("trace", []) + [
            {
                "node": "rewrite",
                "attempt": state.get("retries", 0) + 1,
                "new_query": new_query,
            }
        ]
        return {
            "current_query": new_query,
            "retries": state.get("retries", 0) + 1,
            "trace": trace,
        }

    def route_after_grade(state: SelfRAGState) -> str:
        if state.get("passed"):
            return "done"
        if state.get("retries", 0) >= max_retries:
            return "done"
        return "retry"

    graph = StateGraph(SelfRAGState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("grade", grade_node)
    graph.add_node("rewrite", rewrite_node)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {"done": END, "retry": "rewrite"},
    )
    graph.add_edge("rewrite", "retrieve")

    return graph.compile()


def run_self_rag(graph, question: str) -> dict:
    """
    Run a compiled self-RAG graph on a question. Prints a node-by-node
    trace as the graph executes so the notebook reader can watch the loop.
    Returns the final state, including the answer, score, retry count, and
    the full trace list.
    """
    initial = {
        "question": question,
        "current_query": question,
        "retries": 0,
        "trace": [],
    }

    print(f"Question: {question}")
    print("=" * 70)

    final_state: dict = {
        "question": question,
        "current_query": question,
        "retries": 0,
    }
    for event in graph.stream(initial):
        for node_name, node_output in event.items():
            trace = node_output.get("trace") or []
            last = trace[-1] if trace else {}
            attempt = last.get("attempt", "?")
            if node_name == "retrieve":
                print(f"[attempt {attempt}] retrieve")
                print(f"  query: {last.get('query', '')}")
                print(f"  top sources: {last.get('top_sources', [])}")
            elif node_name == "generate":
                answer = last.get("answer", "")
                preview = answer if len(answer) < 220 else answer[:217] + "..."
                print(f"[attempt {attempt}] generate")
                print(f"  answer: {preview}")
            elif node_name == "grade":
                verdict = "PASS" if last.get("passed") else "FAIL"
                print(
                    f"[attempt {attempt}] grade -> score {last.get('score')} [{verdict}]"
                )
                print(f"  reason: {last.get('reason', '')}")
            elif node_name == "rewrite":
                print(f"[attempt {attempt}] rewrite")
                print(f"  new query: {last.get('new_query', '')}")
            print()
            final_state.update(node_output)

    print("=" * 70)
    print(f"Final answer: {final_state.get('answer', '')}")
    print(
        f"Final score: {final_state.get('score')}  "
        f"retries used: {final_state.get('retries', 0)}"
    )
    return final_state
# === end Chapter 7 helpers ===
