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
