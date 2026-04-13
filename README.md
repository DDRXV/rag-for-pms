# RAG for PMs

Hands-on Google Colab notebooks that pair with the Production RAG video series. Built for product managers who want to understand how real RAG systems work, not just watch someone else build them.

Every chapter is one Colab notebook. Open it, run the cells top to bottom, change a parameter, rerun, watch what happens. No local setup. No dev environment. Click and go.

## What you need

| | |
|---|---|
| **Time per chapter** | About 15 minutes. Chapter 6 (RAGAS evaluation) takes about 20 because the grader is slow. |
| **Total API cost** | Under 70 cents for the whole series. OpenAI gives new accounts $5 of free credit. |
| **Background needed** | None. If you can change a number in a cell and press Run, you can do this. |
| **Python skills needed** | None. Every cell is written so a non-dev can read the narration, run it, and understand the result. |
| **Local install needed** | None. Colab runs everything in the browser. |
| **Accounts to create** | OpenAI (five dollars free credit), LangSmith (free tier), Cohere (free tier). One signup each. Twenty minutes total, once. |

## Start here

If you have never opened one of these, begin with **[Chapter 1: Chunking](https://colab.research.google.com/github/DDRXV/rag-for-pms/blob/main/chapters/01_chunking.ipynb)**. It walks you through setting up your three API keys in Colab secrets, then runs a chunking experiment where one number change flips the answer. After Chapter 1 every other chapter assumes the keys are already in place and picks them up automatically.

Every chapter is independent. You can jump straight to Chapter 5 on Re-ranking if that is what you care about. The keys and corpus are shared, so nothing breaks.

## The chapters

Open in Colab with one click. Each notebook runs against the same SkillAgents AI corpus so the story stays continuous.

| # | Chapter | Open in Colab | What you will build |
|---|---|---|---|
| 01 | **Chunking** | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DDRXV/rag-for-pms/blob/main/chapters/01_chunking.ipynb) | Split a refund policy two ways. One number change drops the retrieval distance from 0.80 to 0.60 and flips the LLM answer from verbose textbook to one clean production sentence. |
| 02 | **Query Translation** | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DDRXV/rag-for-pms/blob/main/chapters/02_query_translation.ipynb) | Take a vague question and rewrite it into three specific ones. Watch retrieval scores go from 1.14 (off topic) to 0.41 (confident) and the answer cover pricing, plans, trial, and refund policy in one paragraph. |
| 03 | **Routing** | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DDRXV/rag-for-pms/blob/main/chapters/03_routing.ipynb) | Send "What is our refund policy" to the doc store, "What was Q3 revenue" to a pandas table, "Who reports to Dan" to an org graph. Same pipeline, three routes, three correct answers. |
| 04 | **Hybrid Search** | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DDRXV/rag-for-pms/blob/main/chapters/04_hybrid_search.ipynb) | Combine BM25 keyword search with vector search so error code E-7829 actually gets found. Watch the alpha sweep flip the answer at each step. |
| 05 | **Re-ranking** | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DDRXV/rag-for-pms/blob/main/chapters/05_reranking.ipynb) | Run Cohere Rerank on top of vector search. Watch the correct refund clause jump from rank 4 to rank 1, and the LLM answer flip from the outdated blog post to the real policy. |
| 06 | **Evaluation** | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DDRXV/rag-for-pms/blob/main/chapters/06_evaluation.ipynb) | Score your pipeline on faithfulness, answer relevance, context precision, and context recall with RAGAS. Watch a deliberately hallucinated answer drop faithfulness from 1.00 to 0.00 while the other three metrics barely move. |
| 07 | **Agentic RAG** | [![Open](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DDRXV/rag-for-pms/blob/main/chapters/07_agentic_rag.ipynb) | Build a Self-RAG loop in LangGraph. A grader node catches the stale marketing blog, rewrites the query, re-retrieves from the authoritative policy, and converges on a grounded answer in one retry. |

## How each chapter works

Every notebook follows the same shape so you do not have to re-learn the layout.

1. **Title and hook.** One paragraph on what you are about to see.
2. **Preview of the payoff.** The before and after stated upfront in plain English.
3. **Install and clone.** One cell, runs once per session, handles both Colab and local Jupyter.
4. **Load your keys.** One line, pulls from Colab secrets.
5. **Core lesson.** Code cells with one tweakable parameter per cell and a plain-English comment next to it.
6. **LLM answer comparison.** See how the answer changes when the retrieval changes.
7. **LangSmith trace.** One link to the visual waterfall of every step.
8. **Monday takeaway.** Three questions you can ask your engineering team on your next RAG review.
9. **Try it yourself.** Three numbered "change this, see what happens" prompts.
10. **Homework.** Two fifteen-minute exercises. One of them asks you to apply the technique to your own company docs.

## Getting your API keys

Three accounts, free tiers cover the entire series, total cost under 70 cents.

1. **OpenAI** at [platform.openai.com](https://platform.openai.com). New accounts get five dollars of credit. You will spend well under a dollar across all seven chapters.
2. **LangSmith** at [smith.langchain.com](https://smith.langchain.com). Free personal tier gives you five thousand traces per month. You will use about fifty.
3. **Cohere** at [dashboard.cohere.com](https://dashboard.cohere.com). Free trial tier gives you one thousand rerank calls per month. You will use maybe twenty.

Chapter 1 walks you through storing these three keys in Colab secrets. You do it once and every other chapter picks them up automatically. No re-entering keys, no hardcoding, no leaks.

## What is in the corpus

Every chapter loads the same set of SkillAgents AI documents. SkillAgents AI is my AI training company for product managers. These documents are small enough to run fast in Colab, but real enough to feel like your own company wrote them.

```
data/skillagents/
├── refund_policy.pdf              Current cohort refund terms. Anchor of Chapters 1, 5, and 7.
├── pricing.pdf                    Plan tiers and prices. Used by the routing and hybrid chapters.
├── product_guide.md               How SkillAgents cohorts actually run. Long form reference.
├── billing_faq.md                 Common billing questions.
├── error_codes.md                 Payment and enrollment error codes. Anchor of the hybrid search chapter.
├── outdated_blog_post.md          Stale marketing post that contradicts the current policy. Anchor of the agentic RAG chapter.
├── ch3_revenue.csv                Quarterly revenue by segment. Used as the SQL-ish route in Chapter 3.
├── ch3_org_chart.py               Mock org chart adjacency dict. Used as the graph route in Chapter 3.
├── ch5_refund_quick_answers.md    Short decoy FAQ that points at the authoritative policy. Used in Chapter 5.
└── ch6_test_set.json              Five golden question-answer pairs for Chapter 6 RAGAS evaluation.
```

The same story runs through every chapter. Each technique you learn gets applied to the same data, so you see how each one changes the same outcome.

## FAQ

**Do I need to know Python?** No. The notebook narration reads like a product doc. You run cells, change one number at a time, and read the output. If you can read a pandas table, you can finish every chapter.

**Will I hit rate limits?** No. The corpus is small. Each chapter makes a handful of OpenAI calls and a handful of Cohere calls. You will not come close to any free tier limit even if you run every chapter twice.

**What if Colab asks me to restart the runtime?** Click Restart, then Run All from the beginning. This sometimes happens after the install cell, once per session. It is not a bug.

**Can I use my own company docs?** Yes. Every chapter ends with a homework prompt that walks you through dropping your own docs into the corpus and running the same lesson on them. Treat it as a take-home exercise for your next RAG project review.

**Do I need a credit card?** No. OpenAI issues the five-dollar trial credit without one. LangSmith and Cohere free tiers do not require a card either.

**What if I already know Python and want to run locally?** See the "Run this locally" section at the bottom. Everything works in a plain Jupyter setup with one `.env.local` file.

**I finished all seven chapters. What next?** Three options, in order of commitment: watch the full video series, join my live cohort on building AI products with Claude Code, or reach out for enterprise work and private cohorts. All three are in the next section.

## Who built this

**Rajesh P**. Former PM at Zynga, Flipkart, and Walmart. Founder of [SkillAgents AI](https://skillagents.ai) and [Codepup AI](https://codepup.ai). I shipped about 80 percent of Codepup AI myself using Claude Code, with no dev background, and now I teach other product managers how to do the same.

You opened this repo to learn RAG. If you want to go further, pick the path that fits where you are right now.

### Free. Watch the full RAG video series

[SkillAgents AI on YouTube](https://www.youtube.com/channel/UCpDUOkGwK2q-NBWZgaH3adw)

Every chapter in this repo pairs with a video that walks through the same concepts visually. Hooks, diagrams, animations. Roughly an hour for the full RAG series. If you liked the notebooks, watching the videos once makes the concepts stick a second time from a different angle. Subscribe and you will also catch the next series on agents, evaluation, and production AI for PMs.

### Paid cohort. Live six weeks, AI Coding for Product Managers

[Join on Maven](https://maven.com/rajeshpeko)

A live cohort on building real products with Claude Code, taught for PMs and non-developers. This is the workflow I used to build Codepup AI as a solo non-dev founder. You bring a product idea. You leave with a working build, a repeatable workflow, and a cohort of peers shipping alongside you.

This cohort is not about RAG. RAG is a narrow technical layer. The cohort covers the bigger picture of shipping an AI product end to end: prompt engineering, tool use, agents, evaluation, and the product decisions that sit around the code. If you liked how these notebooks are structured and you want a live version focused on product building, this is where to go next.

### Limited engagements. Enterprise projects and private cohorts

[Rajesh on LinkedIn](https://www.linkedin.com/in/rajeshpeko/)

Two things I take on directly, in small numbers each quarter.

**Enterprise projects.** Paid engagements to help a product team ship their first production RAG, agentic, or AI-native feature, working alongside me and your engineers for a few weeks. Good fit if your company is trying to turn "we should use AI somewhere" into something real in customer hands.

**Private cohorts.** If your company, team, or community wants a dedicated live workshop on any AI or product management topic (RAG, agents, evaluation, shipping AI features, AI for product discovery, AI for PM workflows, building with Claude Code), reach out. I run private cohorts on request for groups of twenty to forty, tailored to your stack and your questions.

For either one, message me on LinkedIn with one paragraph on what you are working on and we can take it from there.

## Run this locally

Colab is the default. If you want faster reruns, persistent state, or a local IDE, here is the path.

### One time setup

```
git clone https://github.com/DDRXV/rag-for-pms
cd rag-for-pms
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10 or newer.

### API keys for local runs

Create a file called `.env.local` at the root of the repo:

```
OPENAI_API_KEY=sk-your-key-here
LANGCHAIN_API_KEY=your-langsmith-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=rag-for-pms
COHERE_API_KEY=your-cohere-key
```

`.env.local` is already in `.gitignore` so it will not get committed. If you do not have a LangSmith or Cohere key, leave them blank. The notebooks still run, just without traces or rerank.

Before running any notebook, load the env file into your shell:

```
set -a && source .env.local && set +a
```

Then launch Jupyter:

```
jupyter notebook chapters/
```

Or open the repo in VS Code, pick the `.venv` kernel, and run notebooks from inside the IDE.

Each chapter starts with a Colab-vs-local check that handles both environments. In Colab it pip installs packages and clones the repo. Locally it skips both. You do not change anything when switching between the two.
