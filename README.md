# RAG for PMs

Hands-on Colab notebooks that pair with the RAG video series on YouTube. Built for product managers who want to understand how production RAG actually works, not just watch someone else build it.

Every chapter is one notebook. Open it in Colab, run the cells top to bottom, change a parameter, rerun, watch what happens. No local setup. No dev environment. Just click and go.

## What you will learn

| Chapter | Video | What you will build |
|---|---|---|
| 01 Chunking | [RAG Chunking Strategies](https://youtube.com) | Split a refund policy two ways. Watch a similarity score go from 0.61 to 0.94 on the same question. |
| 02 Query Translation | [RAG Query Translation](https://youtube.com) | Take a vague question and rewrite it into three specific ones. See retrieval quality jump from useless to production ready. |
| 03 Routing | [RAG Routing](https://youtube.com) | Send "What is our refund policy" to the doc store and "What was Q3 revenue" to a SQL table. Same system, two different paths. |
| 04 Hybrid Search | [Hybrid Search for RAG](https://youtube.com) | Combine keyword search and vector search so error code E-4012 actually gets found. |
| 05 Re-ranking | [RAG Re-ranking](https://youtube.com) | Run a Cohere re-ranker on top of vector search. Watch the right doc jump from rank four to rank one. |
| 06 Evaluation | [How to Evaluate RAG](https://youtube.com) | Score your pipeline on faithfulness, answer relevance, and context recall with RAGAS. |
| 07 Agentic RAG | [Agentic RAG Explained](https://youtube.com) | Build a Self-RAG loop with LangGraph that checks its own answer and retries when it is wrong. |

## How to run a chapter

1. Click the Colab badge next to the chapter you want.
2. Copy the notebook to your own Drive (File, Save a copy in Drive).
3. Add three API keys to Colab secrets (explained in Chapter 1, takes five minutes the first time).
4. Run all cells. Read the output. Change the highlighted parameter. Run again.

That is the whole workflow. You never touch a terminal. You never install Python. The only thing you type is the parameter you are changing.

## Getting your API keys

You need three keys. All three have free tiers that cover the entire series.

1. **OpenAI** at [platform.openai.com](https://platform.openai.com). New accounts get five dollars of free credit. The whole series uses less than two dollars.
2. **LangSmith** at [smith.langchain.com](https://smith.langchain.com). Free personal tier gives you five thousand traces per month. You will use maybe fifty.
3. **Cohere** at [dashboard.cohere.com](https://dashboard.cohere.com). Free trial tier gives you one thousand rerank calls per month. You will use maybe twenty.

Inside each notebook, Chapter 1 walks you through storing these keys in Colab secrets once. After that every chapter picks them up automatically.

## What is in the corpus

Every chapter loads the same set of SkillAgents AI documents. SkillAgents AI is my AI training company for product managers. The docs are real enough to feel like your own company would write them, but small enough to run fast in Colab.

```
data/skillagents/
├── refund_policy.pdf        The refund terms for cohort enrollments. This is the doc used in the chunking and re-ranking lessons.
├── pricing.pdf              Plan tiers and prices. Used in routing and query construction lessons.
├── product_guide.md         How SkillAgents cohorts actually work. Long-form reference.
├── billing_faq.md           Common billing questions and answers.
├── error_codes.md           Payment and enrollment error codes. Used in the hybrid search lesson.
└── outdated_blog_post.md    A two year old announcement that contradicts the current policy. Used in the agentic RAG lesson.
```

Six docs. Same story in every chapter. Each technique you learn gets applied to the same underlying data, so you see how each technique changes the same outcome.

## Who built this

Rajesh P. Founder of SkillAgents AI and Codepup AI. Former PM at Zynga, Flipkart, and Walmart. This series is part of my Maven cohort "AI Coding for Product Managers".

If you want to go deeper than notebooks, the cohort walks through building RAG inside a real product using Claude Code. Details at [skillagents.ai](https://skillagents.ai).

## Want to run this locally

You do not have to, but if you want to:

```
git clone https://github.com/DDRXV/rag-for-pms
cd rag-for-pms
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook chapters/
```

The notebooks work the same way locally. You still need the three API keys, stored in a `.env` file instead of Colab secrets.
