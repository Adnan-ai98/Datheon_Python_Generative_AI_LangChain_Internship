Document Intelligence RAG

A Retrieval-Augmented Generation (RAG) app that answers questions strictly from your own PDF and text documents, with source file and page citations. Upload documents, tick the ones you want to search, and chat with them.

<!-- Add a screenshot of the app here: ![App screenshot](docs/screenshot.png) -->
Features
PDF and TXT support with page-level metadata
Strict answers: the model only uses the retrieved context and replies "I could not find this information in the selected documents." when the answer is not there
Source citations: every answer lists the file and page numbers it used, plus a "Show sources and relevance scores" panel
Select specific documents: tick only the documents you want to search
Persistent library: uploaded documents are saved and re-indexed automatically, so they are still there after a restart
Delete documents from the sidebar (removes the file and its vectors)
Follow-up questions such as "why does this architecture have the highest accuracy?" are understood from the chat history
Hybrid retrieval: vector similarity + BM25 keyword search, fused with Reciprocal Rank Fusion, run separately on each selected document
Model fallback: if one Gemini model fails or times out, the next one is tried automatically
Debug panel that shows exactly which chunks were retrieved and their scores
How it works
PDF / TXT
   |
   v
PyPDFLoader / TextLoader  ->  RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
   |
   v
HuggingFace embeddings (all-MiniLM-L6-v2, runs locally)  ->  ChromaDB (persistent)
                                                                |
Question -> (optional follow-up rewrite) -> hybrid search <-----+
                 vector similarity + BM25 keyword, per selected document
   |
   v
Strict prompt + numbered context chunks  ->  Gemini  ->  Answer + SOURCES
   |
   v
Streamlit UI: answer, cited files and pages, relevance scores

The retrieval and generation steps are written directly with LangChain components (retriever data + prompt + chat model) instead of the RetrievalQA class, which moved to langchain-classic in LangChain 1.x.

Tech stack
Part	Technology
Orchestration	LangChain
Document loading	PyPDFLoader, TextLoader
Chunking	RecursiveCharacterTextSplitter
Embeddings	sentence-transformers all-MiniLM-L6-v2 (local, free)
Vector store	ChromaDB (persistent)
Keyword search	BM25 (own implementation in src/hybrid.py)
LLM	Google Gemini via langchain-google-genai
UI	Streamlit
Project structure
document-intelligence-rag/
├── app.py                  # Streamlit UI
├── config.py               # Settings (paths, models, chunking, retrieval)
├── debug_retrieval.py      # Terminal tool to inspect retrieval without the LLM
├── requirements.txt
├── .env.example            # Copy to .env and add your API key
├── data/                   # Uploaded documents are saved here
├── chroma_db/              # Persistent vector database
└── src/
    ├── document_loader.py  # Load PDF/TXT, 1-based page numbers
    ├── text_splitter.py    # Recursive chunking
    ├── embeddings.py       # Local embedding model
    ├── vector_store.py     # ChromaDB + hybrid search
    ├── hybrid.py           # BM25 and rank fusion
    └── rag_pipeline.py     # Prompting, model fallback, citations
Getting started
1. Clone and create a virtual environment
bash
git clone https://github.com/<your-username>/document-intelligence-rag.git
cd document-intelligence-rag

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

Python 3.12 is recommended. Very new Python versions may not be supported by ChromaDB yet.

2. Install dependencies
bash
pip install -r requirements.txt
3. Add your Gemini API key

Get a free key from Google AI Studio, then:

bash
# Windows
copy .env.example .env
# macOS / Linux
cp .env.example .env

Open .env and set:

GEMINI_API_KEY=your_gemini_api_key_here
4. Run
bash
streamlit run app.py

Open http://localhost:8501, upload one or more PDF/TXT files in the sidebar, and start asking questions. The first run downloads the embedding model, so it takes a little longer.

Usage
Upload documents in the sidebar. They are saved to data/ and indexed automatically.
Tick the documents you want to search (all are ticked by default).
Ask a question. The answer appears with the cited files and pages.
Open Show sources and relevance scores to see the exact chunks used.
Use New Chat to clear the conversation, or the ✕ button to delete a document.

Example questions for the paper Attention Is All You Need:

What BLEU score did the Transformer achieve on WMT 2014 English-to-German?
How many attention heads are used, and what is d_k?
Why are dot products scaled by 1/sqrt(d_k)?

Questions that are not answered by the selected documents (for example "What is the capital of France?") return the "could not find" message instead of a guess.

Configuration

Environment variables (.env):

Variable	Default	Description
GEMINI_API_KEY	none	Gemini API key (GOOGLE_API_KEY also works)
LLM_MODELS	gemini-3.5-flash-lite,gemini-3.5-flash,gemini-3.6-flash,gemini-3.1-flash-lite	Comma-separated model chain, tried in order
REQUEST_TIMEOUT	15	Seconds to wait for one model before falling back
MODEL_COOLDOWN	300	Seconds a failed model is tried last
THINKING_LEVEL	low	Gemini thinking level, lower is faster. Empty to disable
EMBEDDING_MODEL	sentence-transformers/all-MiniLM-L6-v2	Embedding model

Constants in config.py: CHUNK_SIZE (1000), CHUNK_OVERLAP (200), TOP_K (5 chunks per document), MAX_CHUNKS (16 total), MIN_RELEVANCE (0.15).

Gemini model names change over time. If you get a "model not found" error, update LLM_MODELS with a current model from the Gemini models page.

Debugging retrieval

If an answer is wrong or missing, open Debug: retrieval details under the answer to see the selected documents, the search query and every retrieved chunk with its score. You can also run retrieval from the terminal without calling the LLM:

bash
python debug_retrieval.py "How long did the big models train?" my_paper.pdf
Troubleshooting
Problem	Fix
"API key not found"	Check that .env is in the project root and contains GEMINI_API_KEY=..., then restart Streamlit
Model 404 NOT_FOUND	Set LLM_MODELS in .env to a model that is available to your key
Code changes have no effect	Fully stop Streamlit (Ctrl+C) and run it again. A browser refresh is not enough
ChromaDB tenant / bindings error	Stop the app, clear the contents of chroma_db/, and use a supported Python version (3.12)
Slow answers	Put a fast model first in LLM_MODELS and keep THINKING_LEVEL=low
Limitations
Scanned PDFs without a text layer are not supported (no OCR)
Text inside images, charts and complex tables may not be extracted cleanly
Answers are only as good as the retrieved chunks. Use the debug panel when a fact is missed
License

Add a license of your choice (for example MIT) before publishing.