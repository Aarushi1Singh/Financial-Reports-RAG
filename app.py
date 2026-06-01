import streamlit as st
import anthropic
import chromadb
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ── clients ──────────────────────────────────────────────
client_anthropic = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
client_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

client_chroma = chromadb.PersistentClient(path="./chroma_db")
collection = client_chroma.get_or_create_collection(
    name="financial_rag",
    metadata={"hnsw:space": "cosine"}
)

# ── core functions (copied from notebook) ────────────────
def embed_text(text):
    response = client_openai.embeddings.create(
        input=text, model="text-embedding-3-small"
    )
    return response.data[0].embedding

def classify_query(question):
    response = client_anthropic.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=10,
        messages=[{"role": "user", "content": f"""Classify the following question into exactly one of these categories:
factual - single metric or fact from one specific bank
comparative - same metric or topic across multiple banks
summary - broad qualitative question about a bank's approach or strategy
out_of_scope - completely unrelated to banking and finance 
              (e.g. sports, weather, cooking)
              NOT out_of_scope: anything that could appear in a bank's annual report
              including leadership, board, strategy, operations, ESG

Respond with one word only: factual, comparative, summary, or out_of_scope

Question: {question}"""}]
    )
    return response.content[0].text.strip().lower()

def hyde_retrieve(question, n_results=5):
    response = client_anthropic.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": f"""Write a one-paragraph excerpt from an Indian bank annual report
that would answer this question: {question}
Use specific financial numbers and banking terminology.
Respond with only the paragraph, no preamble."""}]
    )
    hyde_embedding = embed_text(response.content[0].text)
    results = collection.query(
        query_embeddings=[hyde_embedding],
        n_results=n_results,
        where={"level": "leaf"},
        include=["documents", "metadatas", "distances"]
    )
    return [{"text": results["documents"][0][i], "metadata": results["metadatas"][0][i]}
            for i in range(len(results["documents"][0]))]

def multi_query_retrieve(question, n_results=10):
    response = client_anthropic.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=200,
        messages=[{"role": "user", "content": f"""Generate 3 different phrasings of this question for document retrieval:
{question}
Each phrasing should use different terminology to maximise retrieval coverage.
Return as a numbered list. No explanation."""}]
    )
    paraphrases = [line.strip().lstrip("123.").strip()
                   for line in response.content[0].text.strip().split("\n") if line.strip()]
    all_chunks = {}
    for paraphrase in paraphrases:
        embedding = embed_text(paraphrase)
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where={"level": {"$in": ["leaf", "summary_l1", "summary_l2"]}},
            include=["documents", "metadatas", "distances"]
        )
        for i in range(len(results["documents"][0])):
            key = results["documents"][0][i][:100]
            if key not in all_chunks:
                all_chunks[key] = {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i]
                }
    return list(all_chunks.values())

def summary_retrieve(question):
    embedding = embed_text(question)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=5,
        where={"level": "summary_l2"},
        include=["documents", "metadatas", "distances"]
    )
    return [{"text": results["documents"][0][i], "metadata": results["metadatas"][0][i]}
            for i in range(len(results["documents"][0]))]

def retrieve_by_type(question, query_type, top_k=5):
    if query_type == "factual":
        return hyde_retrieve(question, n_results=top_k)
    elif query_type == "comparative":
        return multi_query_retrieve(question, n_results=top_k)
    elif query_type == "summary":
        return summary_retrieve(question)
    return []

def grade_context(question, context_text):
    response = client_anthropic.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=100,
        messages=[{"role": "user", "content": f"""Given this question: {question}
And this retrieved context: {context_text[:6000]}
Rate how well the context answers the question on a scale of 0.0 to 1.0.
0.0 = completely irrelevant. 0.5 = partially answers. 0.8 = mostly answers. 1.0 = fully answers.
Important: if context contains relevant financial data even if incomplete, score at least 0.6.
Respond with only: score|one-sentence rationale"""}]
    )
    parts = response.content[0].text.strip().split("|")
    return float(parts[0].strip()), parts[1].strip() if len(parts) > 1 else ""

def generate_answer(question, chunks):
    context = "\n\n".join([
        f"[{c['metadata']['bank_name']}, Page {c['metadata'].get('page_number', 'N/A')}, {c['metadata']['level']}]\n{c['text']}"
        for c in chunks
    ])
    response = client_anthropic.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": f"""You are a financial analyst assistant. Answer only based on the provided context.
For every fact or number you state, cite the source as [Bank Name, Page X].
If the context does not contain the answer, say: 'Not found in provided reports.'

Context:
{context}

Question: {question}"""}]
    )
    return response.content[0].text

def query_with_grading(question):
    query_type = classify_query(question)
    if query_type == "out_of_scope":
        return {"answer": "Not found in provided reports.", "confidence": 0.0,
                "query_type": query_type, "sources": []}
    top_k = 10 if query_type == "comparative" else 5
    chunks = retrieve_by_type(question, query_type, top_k=top_k)
    context_text = " ".join([c["text"] for c in chunks])
    score, rationale = grade_context(question, context_text)
    if score < 0.4:
        chunks = retrieve_by_type(question, query_type, top_k=top_k + 8)
        context_text = " ".join([c["text"] for c in chunks])
        score, rationale = grade_context(question, context_text)
    if score < 0.3:
        return {"answer": "Insufficient context found in provided reports.",
                "confidence": score, "query_type": query_type, "sources": []}
    answer = generate_answer(question, chunks)
    sources = list({(c["metadata"]["bank_name"], c["metadata"].get("page_number", "N/A"))
                    for c in chunks})
    return {"answer": answer, "confidence": score, "query_type": query_type, "sources": sources}

# ── streamlit UI ─────────────────────────────────────────
st.set_page_config(page_title="Financial RAG Analyst", page_icon="🏦", layout="wide")

st.title("🏦 Financial RAG Analyst")
st.caption("Ask questions across 5 Indian bank FY25 annual reports")

# sidebar
with st.sidebar:
    st.header("Example Queries")
    st.markdown("**Factual**")
    if st.button("HDFC Bank NPA ratio FY25?"):
        st.session_state.example = "What was HDFC Bank's gross NPA ratio in FY25?"
    if st.button("SBI net interest margin?"):
        st.session_state.example = "What was SBI's net interest margin in FY25?"
    
    st.markdown("**Comparative**")
    if st.button("Compare CAR across all banks"):
        st.session_state.example = "Compare capital adequacy ratios across all 5 banks."
    if st.button("Which bank had highest ROE?"):
        st.session_state.example = "Which bank had the highest return on equity in FY25?"
    
    st.markdown("**Summary**")
    if st.button("ICICI risk management"):
        st.session_state.example = "Summarise ICICI Bank's risk management approach."
    if st.button("HDFC strategy FY25"):
        st.session_state.example = "What was HDFC Bank's strategy in FY25?"
    
    st.divider()
    st.caption("5 banks · FY25 · ~11,600 chunks indexed")

# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            col1, col2 = st.columns(2)
            with col1:
                st.caption(f"Query type: `{message['query_type']}`")
            with col2:
                confidence = message["confidence"]
                color = "🟢" if confidence >= 0.8 else "🟡" if confidence >= 0.6 else "🔴"
                st.caption(f"Confidence: {color} {confidence:.2f}")
            if message["sources"]:
                with st.expander("📄 Sources"):
                    for bank, page in sorted(message["sources"]):
                        st.caption(f"• {bank} — Page {page}")

# handle example button clicks
if "example" in st.session_state:
    prompt = st.session_state.example
    del st.session_state.example
else:
    prompt = None

# chat input
user_input = st.chat_input("Ask about Indian banks...")
if user_input:
    prompt = user_input

if prompt:
    # show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = query_with_grading(prompt)
        
        st.markdown(result["answer"])
        
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"Query type: `{result['query_type']}`")
        with col2:
            confidence = result["confidence"]
            color = "🟢" if confidence >= 0.8 else "🟡" if confidence >= 0.6 else "🔴"
            st.caption(f"Confidence: {color} {confidence:.2f}")
        
        if result["sources"]:
            with st.expander("📄 Sources"):
                for bank, page in sorted(result["sources"]):
                    st.caption(f"• {bank} — Page {page}")
    
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "confidence": result["confidence"],
        "query_type": result["query_type"]
    })