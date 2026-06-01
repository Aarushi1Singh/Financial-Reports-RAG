# Financial RAG Analyst

An Advanced RAG system for querying FY25 annual reports of 5 major Indian banks — HDFC, ICICI, SBI, Axis, and Kotak — using natural language.

## Features
- **RAPTOR indexing** — hierarchical summaries for cross-bank queries
- **HyDE retrieval** — hypothetical document embedding for factual queries
- **Multi-Query expansion** — paraphrase-based retrieval for comparative queries
- **Logical LLM router** — classifies queries into factual / comparative / summary / out_of_scope
- **Adaptive RAG** — self-grading loop with re-retrieval
- **Streamlit chat UI** — confidence score, citations, query type label

## Tech Stack
- LangChain, ChromaDB, OpenAI Embeddings
- Anthropic Claude API (claude-sonnet-4-5)
- Streamlit

## Setup

1. Clone the repo
git clone https://github.com/yourusername/financial-rag-analyst.git
cd financial-rag-analyst

2. Install dependencies
pip install -r requirements.txt

3. Create `.env` file
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key

4. Download FY25 annual reports for all 5 banks into `data/pdfs/`
- HDFC Bank — investor.hdfcbank.com
- ICICI Bank — icicibank.com/investor-relations
- SBI — sbi.co.in
- Axis Bank — axisbank.com/investor-relations
- Kotak Mahindra Bank — kotak.com/investor-relations

5. Run indexing pipeline (once)
jupyter notebook annual_report_RAG.ipynb
Run all cells in order — this parses, chunks, embeds, and indexes all documents into ChromaDB.

6. Launch the app
streamlit run app.py

## Architecture

### Indexing Phase (runs once)
PDF parsing → Leaf chunking → RAPTOR clustering & summarisation → Embedding → ChromaDB

### Query Phase (every question)
User question → LLM Router → HyDE / Multi-Query / RAPTOR L2 → Self-grading → Answer generation → Streamlit UI

## Data
5 Indian bank FY25 annual reports (~2,460 pages, ~11,600 indexed nodes)

