import streamlit as st
import anthropic
from pinecone import Pinecone
from openai import OpenAI
from tavily import TavilyClient
import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv(override=True)

client_anthropic = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

# ── supabase client ──────────────────────────────────────
supabase_client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX", "financial-rag"))

st.set_page_config(
    page_title="Financial RAG Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer { visibility: hidden; }
header { visibility: visible !important; }
.block-container { padding: 0 2rem 8rem 2rem !important; max-width: 860px !important; margin: 0 auto !important; }
[data-testid="stAppViewContainer"] { background: #0C0D0E; }
[data-testid="stHeader"] { background: transparent !important; }

/* ── hide sidebar collapse/expand arrows — sidebar stays fixed ── */
[data-testid="stSidebarCollapseButton"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }

/* ── sidebar ── */
.sidebar-logo-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.1rem;
    margin-top: 0.2rem;
}
.sidebar-arrow {
    font-size: 0.75rem;
    color: #7B9EC4;
    cursor: pointer;
    opacity: 0.8;
    font-family: sans-serif;
}

[data-testid="stSidebar"] {
    background: #101214 !important;
    border-right: 1px solid #1C1E21 !important;
    min-width: 230px !important;
    max-width: 230px !important;
}
[data-testid="stSidebarContent"] { padding: 0.75rem 0.9rem !important; }
[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; margin-top: 0 !important; }
[data-testid="stSidebarHeader"] { display: none !important; min-height: 0 !important; height: 0 !important; padding: 0 !important; }
[data-testid="stSidebar"] .stMarkdown { margin: 0 !important; padding: 0 !important; }
[data-testid="stSidebar"] .stMarkdown p { margin: 0 !important; padding: 0 !important; }
[data-testid="stSidebar"] .element-container { margin: 0 !important; padding: 0 !important; min-height: 0 !important; }
[data-testid="stSidebar"] .stButton { margin: 0 !important; padding: 0 !important; }
[data-testid="stSidebar"] .element-container:has(.stButton) { margin: 0 !important; padding: 0 !important; }
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] { gap: 0 !important; }
[data-testid="stSidebar"] [data-testid="stVerticalBlockBorderWrapper"] { padding: 0 !important; margin: 0 !important; }

.sidebar-logo {
    font-family: 'DM Serif Display', serif;
    font-size: 1.05rem;
    color: #D4DCEB;
    letter-spacing: 0.01em;
    margin-bottom: 0.1rem;
}
.sidebar-sub {
    font-size: 0.63rem;
    color: #4A5570;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.sidebar-divider {
    border: none;
    border-top: 1px solid #1C1E21;
    margin: 0.4rem 0;
}
.sidebar-section {
    font-size: 0.63rem;
    font-weight: 500;
    color: #7B8A9E;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.55rem;
    margin-top: 0.3rem;
}
.sidebar-body {
    font-size: 0.72rem;
    color: #7B8A9E;
    line-height: 1.6;
    margin-bottom: 0.25rem;
}
.sidebar-tag {
    display: inline-block;
    font-size: 0.58rem;
    font-family: 'DM Mono', monospace;
    color: #7B9EC4;
    border: 1px solid #1E2E40;
    border-radius: 3px;
    padding: 2px 5px;
    margin: 2px 2px 2px 0;
}
.sidebar-tree {
    margin-bottom: 0.25rem;
}
.tree-parent {
    font-size: 0.63rem;
    font-weight: 500;
    color: #7B8A9E;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 0.1rem;
    margin-top: 0.2rem;
    padding-left: 0;
}
.tree-children {
    font-size: 0.67rem;
    font-family: 'DM Mono', monospace;
    color: #7B9EC4;
    padding-left: 0.6rem;
    border-left: 1px solid #1E2E40;
    line-height: 1.4;
    margin-bottom: 0.1rem;
}
.sidebar-stat {
    font-size: 0.68rem;
    color: #7B8A9E;
    margin-bottom: 0.2rem;
    display: flex;
    justify-content: space-between;
}
.sidebar-stat span {
    font-family: 'DM Mono', monospace;
    color: #7B9EC4;
    font-size: 0.65rem;
}

/* ── sidebar buttons ── */
[data-testid="stSidebar"] .stButton:first-of-type > button {
    margin-top: 0.4rem !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: transparent !important;
    border: 1px solid #1C1E21 !important;
    color: #5A6070 !important;
    font-size: 0.7rem !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.3rem 0.55rem !important;
    border-radius: 4px !important;
    text-align: left !important;
    width: 100% !important;
    margin-bottom: 0.15rem !important;
    transition: all 0.15s ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    border-color: #3C4A60 !important;
    color: #D4DCEB !important;
    background: #131618 !important;
}

/* ── main area ── */
.main-area {
    padding: 2rem 5rem 10rem 5rem;
    max-width: 860px;
    margin: 0 auto;
}
.page-hero {
    text-align: center;
    padding: 2.5rem 1rem 2rem 1rem;
    border-bottom: 1px solid #1C1E21;
    margin-bottom: 2rem;
}
.chat-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.6rem;
    color: #D4DCEB;
    margin-bottom: 0.35rem;
    font-style: italic;
}
.chat-subheader {
    font-size: 0.75rem;
    font-weight: 600;
    color: #5A6A80;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}

/* ── messages ── */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin-bottom: 1.5rem;
    margin-left: 1rem;
    margin-right: 1rem;
}
.msg-user-bubble {
    background: #141618;
    border: 1px solid #1E2126;
    border-radius: 12px 12px 2px 12px;
    padding: 0.7rem 1rem;
    max-width: 70%;
    font-size: 0.84rem;
    color: #D4DCEB;
    line-height: 1.6;
}
.msg-assistant {
    display: flex;
    gap: 0.7rem;
    margin-bottom: 1.75rem;
    margin-left: 1rem;
    margin-right: 1rem;
    align-items: flex-start;
}
.msg-avatar {
    width: 26px;
    height: 26px;
    border-radius: 5px;
    background: #141618;
    border: 1px solid #1E2126;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    color: #7B9EC4;
    font-family: 'DM Serif Display', monospace;
    flex-shrink: 0;
    margin-top: 3px;
    font-style: italic;
}
.msg-assistant-content { flex: 1; max-width: 88%; }
.msg-assistant-bubble {
    font-size: 0.84rem;
    color: #A0AABB;
    line-height: 1.8;
}
.msg-assistant-bubble strong { color: #D4DCEB; font-weight: 500; }
.msg-assistant-bubble h1,.msg-assistant-bubble h2,.msg-assistant-bubble h3 {
    font-family: 'DM Serif Display', serif;
    color: #D4DCEB;
    font-size: 0.92rem;
    margin: 1rem 0 0.4rem;
}
.msg-assistant-bubble ul,.msg-assistant-bubble ol { padding-left: 1.2rem; margin: 0.4rem 0; }
.msg-assistant-bubble li { margin-bottom: 0.25rem; }
.msg-assistant-bubble table { width: 100%; border-collapse: collapse; margin: 0.6rem 0; font-size: 0.78rem; }
.msg-assistant-bubble th {
    text-align: left; padding: 0.35rem 0.7rem;
    border-bottom: 1px solid #1C1E21;
    color: #5A6070; font-weight: 500;
    font-size: 0.66rem; letter-spacing: 0.07em; text-transform: uppercase;
}
.msg-assistant-bubble td { padding: 0.35rem 0.7rem; border-bottom: 1px solid #111315; color: #A0AABB; }

/* ── meta ── */
.meta-row {
    display: flex; align-items: center; gap: 0.55rem;
    margin-top: 0.55rem; flex-wrap: wrap;
}
.meta-chip {
    font-size: 0.6rem; font-family: 'DM Mono', monospace;
    padding: 2px 6px; border-radius: 3px;
    letter-spacing: 0.04em; border: 1px solid;
}
.chip-factual     { color: #7B9EC4; border-color: #1E2E40; background: #0D1520; }
.chip-comparative { color: #9E7BC4; border-color: #2A1E40; background: #150D20; }
.chip-summary     { color: #9EC47B; border-color: #2A401E; background: #151D0D; }
.chip-out_of_scope,.chip-web_search { color: #C4A87B; border-color: #402E1E; background: #20150D; }
.confidence-dot { width: 5px; height: 5px; border-radius: 50%; display: inline-block; }
.conf-high { background: #7B9EC4; }
.conf-mid  { background: #9EC47B; }
.conf-low  { background: #5A6070; }
.conf-text { font-size: 0.6rem; font-family: 'DM Mono', monospace; color: #3C4250; }

/* ── expander ── */
[data-testid="stExpander"] {
    background: transparent !important;
    border: 1px solid #1C1E21 !important;
    border-radius: 5px !important;
    margin-top: 0.45rem !important;
}
[data-testid="stExpander"] summary {
    font-size: 0.65rem !important;
    color: #3C4250 !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.06em !important;
    text-transform: uppercase !important;
}
[data-testid="stExpander"] p {
    font-size: 0.7rem !important;
    font-family: 'DM Mono', monospace !important;
    color: #3C4250 !important;
    line-height: 1.8 !important;
}

/* ── input ── */
[data-testid="stChatInput"] {
    max-width: 860px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}
[data-testid="stChatInput"] textarea {
    background: #101214 !important;
    border: 1px solid #1C1E21 !important;
    border-radius: 8px !important;
    color: #D4DCEB !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.84rem !important;
    padding: 0.75rem 1rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #3C4A60 !important;
    box-shadow: none !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #3A3D42 !important; }
[data-testid="stChatInputSubmitButton"] {
    background: #141618 !important;
    border: 1px solid #252729 !important;
    border-radius: 6px !important;
}

/* ── empty state ── */
.empty-state { display: none; }
</style>
""", unsafe_allow_html=True)

# ── core functions ────────────────────────────────────────
def embed_text(text):
    r = client_openai.embeddings.create(input=text, model="text-embedding-3-small")
    return r.data[0].embedding

def classify_query(question):
    time_keywords = ["current", "today", "now", "latest", "right now", "as of today"]
    if any(kw in question.lower() for kw in time_keywords):
        return "out_of_scope"

    # ranking/comparison questions always route to comparative — bypass LLM
    ranking_keywords = ["highest", "lowest", "best", "worst", "most", "least", "which bank",
                        "better", "worse", "outperform", "compare", "comparison", "perform"]
    if any(kw in question.lower() for kw in ranking_keywords):
        return "comparative"

    r = client_anthropic.messages.create(
        model="claude-sonnet-4-5", max_tokens=10, temperature=0,
        messages=[{"role": "user", "content": f"""Classify into one of: factual / comparative / summary / out_of_scope

The 5 banks in scope are: HDFC Bank, ICICI Bank, SBI, Axis Bank, Kotak Mahindra Bank.
Their FY25 annual reports are fully indexed. Any question answerable from these reports is IN scope.

factual     - specific metric or fact from one named bank's FY25 annual report
comparative - comparing any metric across multiple banks, OR "which bank" questions on any metric
summary     - broad qualitative question about a bank's strategy, approach, or risk management
out_of_scope - ONLY if it requires live/real-time data (stock price, today's rate) OR is completely
               unrelated to Indian banking (e.g. "what is the capital of France")

CRITICAL: "which bank performed better on X", "which bank had the highest/lowest X",
          "best bank for X", "which bank outperformed on loans" are ALL comparative — never out_of_scope.

One word only. Question: {question}"""}]
    )
    return r.content[0].text.strip().lower()

def hyde_retrieve(question, n_results=5):
    r = client_anthropic.messages.create(
        model="claude-sonnet-4-5", max_tokens=300,
        messages=[{"role": "user", "content": f"Write a one-paragraph Indian bank annual report excerpt answering: {question}\nUse specific numbers and banking terminology. Paragraph only."}]
    )
    emb = embed_text(r.content[0].text)
    res = index.query(vector=emb, top_k=n_results, filter={"level": {"$eq": "leaf"}}, include_metadata=True)
    return [{"text": m.metadata["text"], "metadata": m.metadata} for m in res.matches]

def multi_query_retrieve(question, n_results=10):
    r = client_anthropic.messages.create(
        model="claude-sonnet-4-5", max_tokens=200, temperature=0,
        messages=[{"role": "user", "content": f"Generate 3 different phrasings of this for document retrieval:\n{question}\nNumbered list only."}]
    )
    paraphrases = [l.strip().lstrip("123.").strip() for l in r.content[0].text.strip().split("\n") if l.strip()]
    seen = {}
    for p in paraphrases:
        res = index.query(vector=embed_text(p), top_k=n_results,
                          filter={"level": {"$in": ["leaf", "summary_l1", "summary_l2"]}},
                          include_metadata=True)
        for m in res.matches:
            k = m.id
            if k not in seen:
                seen[k] = {"text": m.metadata["text"], "metadata": m.metadata, "distance": 1 - m.score}
    return list(seen.values())

def summary_retrieve(question):
    # query L2 first (bank-level summaries)
    res_l2 = index.query(vector=embed_text(question), top_k=3,
                         filter={"level": {"$eq": "summary_l2"}}, include_metadata=True)
    chunks = [{"text": m.metadata["text"], "metadata": m.metadata} for m in res_l2.matches]
    seen = {m.metadata["text"][:100] for m in res_l2.matches}

    # supplement with top L1 nodes — L2 may not cover specific topics
    # (e.g. risk management may land in its own L1 cluster but not dominate L2)
    res_l1 = index.query(vector=embed_text(question), top_k=4,
                         filter={"level": {"$eq": "summary_l1"}}, include_metadata=True)
    for m in res_l1.matches:
        if m.metadata["text"][:100] not in seen:
            chunks.append({"text": m.metadata["text"], "metadata": m.metadata})
            seen.add(m.metadata["text"][:100])

    return chunks

def retrieve_by_type(question, query_type, top_k=5):
    if query_type == "factual": return hyde_retrieve(question, n_results=top_k)
    elif query_type == "comparative": return multi_query_retrieve(question, n_results=top_k)
    elif query_type == "summary": return summary_retrieve(question)
    return []

def grade_context(question, context_text):
    r = client_anthropic.messages.create(
        model="claude-sonnet-4-5", max_tokens=100, temperature=0,
        messages=[{"role": "user", "content": f"""Question: {question}
        Context: {context_text[:6000]}

        Does the context contain a specific answer to this question?
        - If the exact metric/figure asked for is present: score 0.8-1.0
        - If related but different metrics are present: score 0.3-0.5
        - If completely irrelevant: score 0.0-0.2

        Respond EXACTLY: 0.85|one sentence only. Nothing else."""}]    )
    import re
    raw = r.content[0].text.strip()
    try:
        if "|" in raw:
            parts = raw.split("|", 1)
            return float(parts[0].strip()), parts[1].strip()
        match = re.search(r"(\d+\.?\d*)", raw)
        return (float(match.group(1)) if match else 0.5), raw
    except Exception:
        return 0.5, "parse error"

def generate_answer(question, chunks):
    context_parts = []
    for chunk in chunks:
        bank = chunk["metadata"]["bank_name"]
        level = chunk["metadata"]["level"]
        page = chunk["metadata"].get("page_number", "N/A")
        if level == "leaf" and str(page) != "N/A":
            citation = f"[{bank}, Page {page}]"
        else:
            citation = f"[{bank}, FY25 Annual Report]"
        context_parts.append(f"{citation}\n{chunk['text']}")
    ctx = "\n\n".join(context_parts)
    r = client_anthropic.messages.create(
        model="claude-sonnet-4-5", max_tokens=1000, temperature=0,
        messages=[{"role": "user", "content": f"""You are a financial analyst assistant. Answer only from the provided context.

Ranking/comparison rule — CRITICAL:
- If the question asks "which bank had the highest/lowest/best X", scan ALL figures for that metric across ALL banks in the context BEFORE writing your answer.
- Determine the correct ranking first, then state it once at the top as your final answer.
- Do NOT state a preliminary answer and then correct it. One conclusion only.

Citation rules — follow exactly:
- Use the citation tag shown before each context block as-is
- For [Bank, Page X] sources: cite as [Bank, Page X]
- For [Bank, FY25 Annual Report] sources: cite as [Bank, FY25 Annual Report]
- Never write Page N/A — if no page number exists use FY25 Annual Report
- One citation per fact. All figures are FY25.
- Never say not found and then quote data.

Context:
{ctx}

Question: {question}"""}]
    )
    return r.content[0].text

# ── bank keyword guard — blocks open web for any bank/finance question ──
BANK_KEYWORDS = [
    "hdfc", "icici", "sbi", "axis", "kotak", "bank", "npa", "gnpa", "nnpa", "car", "crar",
    "roe", "nim", "roa", "capital adequacy", "loan", "credit", "deposit", "interest",
    "margin", "ratio", "perform", "return on", "net interest", "gross", "asset quality",
    "liquidity", "revenue", "profit", "earnings", "balance sheet", "tier", "advance",
    "borrowing", "provision", "slippage", "pcr", "coverage", "risk management",
    "risk approach", "strategy", "governance", "esg", "sustainability"
]

def is_bank_question(question):
    q = question.lower()
    return any(kw in q for kw in BANK_KEYWORDS)

def web_search_fallback(question):
    # bank-related questions must never hit open web — return clean not-found instead
    if is_bank_question(question):
        return {
            "answer": (
                "The relevant data was not found in the FY25 annual reports of the 5 indexed banks "
                "(HDFC Bank, ICICI Bank, SBI, Axis Bank, Kotak Mahindra Bank). "
                "Try rephrasing your question, or ask about a specific metric from one of these banks."
            ),
            "confidence": 0.0,
            "query_type": "out_of_scope",
            "sources": []
        }
    # only genuinely external queries reach here (e.g. current RBI repo rate, macro news)
    res = tavily_client.search(query=question + " India 2025", search_depth="advanced", max_results=3)
    ctx = ""
    sources = []
    for r in res["results"]:
        ctx += f"[{r['url']}]\n{r['content']}\n\n"
        sources.append(("Web", r['url'][:50]))
    r = client_anthropic.messages.create(
        model="claude-sonnet-4-5", max_tokens=800,
        messages=[{"role": "user", "content": f"Answer using web results. Year is 2026. Use most recent data. Cite URLs.\n\n{ctx}\nQuestion: {question}"}]
    )
    return {"answer": r.content[0].text, "confidence": 0.5, "query_type": "web_search", "sources": sources}

def log_query(question, query_type, confidence, answer, web_search_used, sources):
    """Log every query to Supabase for analytics and A/B testing."""
    try:
        sources_str = ", ".join([f"{b} pg{p}" for b, p in sources if str(p) != "N/A"]) if sources else ""
        supabase_client.table("chat_history").insert({
            "question": question,
            "query_type": query_type,
            "confidence": float(confidence),
            "answer": answer[:2000],
            "web_search_used": query_type == "web_search",
            "sources": sources_str
        }).execute()
    except Exception as e:
        pass  # never let logging break the app

def query_with_grading(question):
    qt = classify_query(question)
    if qt == "out_of_scope":
        result = web_search_fallback(question)
        log_query(question, result["query_type"], result["confidence"], result["answer"], True, result["sources"])
        return result
    top_k = 15 if qt == "comparative" else 8
    chunks = retrieve_by_type(question, qt, top_k=top_k + 10)
    # comparative: keep retrieval-score order so highest-relevance chunks lead
    # factual/summary: sort by bank name for consistent citation ordering
    if qt != "comparative":
        chunks = sorted(chunks, key=lambda x: x["metadata"]["bank_name"])
    ctx = " ".join([c["text"] for c in chunks])
    score, _ = grade_context(question, ctx)
    if score < 0.4:
        chunks = retrieve_by_type(question, qt, top_k=top_k + 8)
        ctx = " ".join([c["text"] for c in chunks])
        score, _ = grade_context(question, ctx)
    if score < 0.3:
        # bank questions: return not-found cleanly — never leak to open web
        if is_bank_question(question):
            result = {
                "answer": (
                    "The relevant data was not found in the FY25 annual reports of the 5 indexed banks "
                    "(HDFC Bank, ICICI Bank, SBI, Axis Bank, Kotak Mahindra Bank). "
                    "Try rephrasing your question, or ask about a specific metric from one of these banks."
                ),
                "confidence": 0.0,
                "query_type": "out_of_scope",
                "sources": []
            }
        else:
            result = web_search_fallback(question)
        log_query(question, result["query_type"], result["confidence"], result["answer"], True, result["sources"])
        return result
    answer = generate_answer(question, chunks)

    # ── catch "not found" answers — only fall back to web if NOT a bank question ──
    not_found_phrases = [
        "not found in", "not mentioned", "not available in",
        "does not include", "not provided in", "cannot find",
        "no information", "not present in"
    ]
    if any(phrase in answer.lower() for phrase in not_found_phrases) and not is_bank_question(question):
        result = web_search_fallback(question)
        log_query(question, result["query_type"], result["confidence"], result["answer"], True, result["sources"])
        return result

    sources = list({(c["metadata"]["bank_name"], c["metadata"].get("page_number", "N/A")) for c in chunks})
    log_query(question, qt, score, answer, False, sources)
    return {"answer": answer, "confidence": score, "query_type": qt, "sources": sources}

# ── sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown('''
    <div class="sidebar-logo-row">
        <span class="sidebar-logo">Financial RAG</span>
        
    </div>
    <div class="sidebar-sub">Annual Report Analysis</div>
    <hr class="sidebar-divider">
    <div class="sidebar-section">About</div>
    <div class="sidebar-body">
        Query FY25 annual reports of 5 Indian banks.
        Answers sourced directly from the reports.
        Falls back to web search when data isn't in the reports.
    </div>
    <div class="sidebar-section">Index</div>
    <div class="sidebar-tree">
        <div class="tree-parent">Banks</div>
        <div class="tree-children">HDFC &nbsp;·&nbsp; ICICI &nbsp;·&nbsp; SBI &nbsp;·&nbsp; Axis &nbsp;·&nbsp; Kotak</div>
        <div class="tree-parent">Period</div>
        <div class="tree-children">FY 2024–25</div>
    </div>
    <hr class="sidebar-divider">
    <div class="sidebar-section" style="margin-bottom: 0.6rem; padding-bottom: 0.3rem;">Try asking</div>
    ''', unsafe_allow_html=True)

    examples = [
        ("HDFC NPA ratio", "What was HDFC Bank's gross NPA ratio in FY25?"),
        ("Compare CAR — all banks", "Compare capital adequacy ratios across all 5 banks."),
        ("SBI net interest margin", "What was SBI's net interest margin in FY25?"),
        ("ICICI risk approach", "Summarise ICICI Bank's risk management approach."),
        ("Highest ROE?", "Which bank had the highest return on equity in FY25?"),
    ]
    for label, query in examples:
        if st.button(label, key=f"ex_{label}"):
            st.session_state.pending = query
            st.rerun()

    st.markdown('''
    <hr class="sidebar-divider">
    <div style="margin-bottom:0.25rem">
        <span class="sidebar-tag">RAPTOR</span>
        <span class="sidebar-tag">HyDE</span>
        <span class="sidebar-tag">Multi-Query</span>
    </div>
    <div>
        <span class="sidebar-tag">LLM Router</span>
        <span class="sidebar-tag">Adaptive RAG</span>
        <span class="sidebar-tag">Web Search</span>
    </div>
    ''', unsafe_allow_html=True)

# ── session state ─────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

prompt = None
if "pending" in st.session_state:
    prompt = st.session_state.pending
    del st.session_state.pending

# ── main layout ───────────────────────────────────────────
st.markdown('<div class="main-area">', unsafe_allow_html=True)
st.markdown('''<div class="page-hero">
    <div class="chat-header">Ask the reports</div>
    <div class="chat-subheader">5 banks &nbsp;·&nbsp; FY25 &nbsp;·&nbsp; HDFC &nbsp;·&nbsp; ICICI &nbsp;·&nbsp; SBI &nbsp;·&nbsp; Axis &nbsp;·&nbsp; Kotak</div>
</div>''', unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="msg-user"><div class="msg-user-bubble">{msg["content"]}</div></div>',
                    unsafe_allow_html=True)
    else:
        qt = msg.get("query_type", "factual")
        conf = msg.get("confidence", 0)
        conf_cls = "conf-high" if conf >= 0.8 else "conf-mid" if conf >= 0.6 else "conf-low"
        with st.container():
            col_avatar, col_body = st.columns([0.04, 0.96])
            with col_avatar:
                st.markdown('', unsafe_allow_html=True)
            with col_body:
                st.markdown(f'<div class="msg-assistant-bubble">', unsafe_allow_html=True)
                st.markdown(msg["content"])
                st.markdown(f'''<div class="meta-row">
                    <span class="meta-chip chip-{qt}">{qt.replace("_"," ")}</span>
                    <span class="confidence-dot {conf_cls}"></span>
                    <span class="conf-text">{conf:.2f}</span>
                </div>''', unsafe_allow_html=True)
                if msg.get("sources"):
                    unique_sources = list({(b, str(p)) for b, p in msg["sources"] if str(p) != "N/A"})
                    unique_sources.sort(key=lambda x: x[0])
                    if unique_sources:
                        with st.expander(f"sources · {len(unique_sources)} referenced"):
                            for bank, page in unique_sources:
                                st.caption(f"{bank}  ·  pg {page}")
        st.markdown('<div style="margin-bottom:1.75rem"></div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ── sidebar-aware chat input positioning ──────────────────
st.markdown("""<script>
(function() {
    function updateInputOffset() {
        var sidebar = window.parent.document.querySelector('[data-testid="stSidebar"]');
        var input   = window.parent.document.querySelector('[data-testid="stChatInput"]');
        if (!sidebar || !input) return;
        var expanded = sidebar.getAttribute('aria-expanded') !== 'false'
                       && !sidebar.style.marginLeft.includes('-');
        input.style.left = expanded ? '230px' : '0px';
    }
    updateInputOffset();
    var obs = new MutationObserver(updateInputOffset);
    obs.observe(window.parent.document.body, { attributes: true, subtree: true, attributeFilter: ['style','aria-expanded'] });
})();
</script>""", unsafe_allow_html=True)

# ── input ─────────────────────────────────────────────────
user_input = st.chat_input("ask a question")
if user_input:
    prompt = user_input

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.spinner(""):
        result = query_with_grading(prompt)
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "confidence": result["confidence"],
        "query_type": result["query_type"]
    })
    st.rerun()
