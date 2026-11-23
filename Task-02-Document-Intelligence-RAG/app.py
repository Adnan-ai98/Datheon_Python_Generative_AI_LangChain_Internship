from collections import defaultdict
from pathlib import Path

import streamlit as st

import config
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="AI-Powered Document Q&A", page_icon="📄", layout="wide")


@st.cache_resource
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()


try:
    rag = get_pipeline()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

st.session_state.setdefault("history", [])
st.session_state.setdefault("uploader_key", 0)
st.session_state.setdefault("flash", None)

# Index any saved document that is missing from the database (runs once per session)
if "synced" not in st.session_state:
    with st.spinner("Loading saved documents..."):
        rag.sync()
    st.session_state.synced = True


def format_cited(sources) -> str:
    pages = defaultdict(set)
    for s in sources:
        pages[s["source"]].add(s["page"])
    return ", ".join(
        f"{name} (p. {', '.join(str(p) for p in sorted(pgs))})"
        for name, pgs in pages.items()
    )


def render_message(msg) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            if msg.get("sources"):
                st.markdown(f"**Sources cited:** {format_cited(msg['sources'])}")
                with st.expander("Show sources and relevance scores"):
                    for s in msg["sources"]:
                        st.markdown(
                            f"**{s['source']}** - page {s['page']} - relevance {s['score']:.0%}"
                        )
                        st.caption(s["snippet"] + "...")
            if msg.get("model"):
                took = f" in {msg['seconds']:.1f}s" if msg.get("seconds") else ""
                st.caption(f"Answered by {msg['model']}{took}")
            if msg.get("notes"):
                st.caption("Fallback used: " + "; ".join(msg["notes"]))
            if msg.get("debug"):
                with st.expander("Debug: retrieval details"):
                    dbg = msg["debug"]
                    st.caption("Selected documents: " + (", ".join(dbg["selected"]) or "none"))
                    st.caption("Search query: " + dbg["query"])
                    for c in dbg["chunks"]:
                        flag = "used" if c["used"] else "below threshold"
                        st.markdown(f"`{c['source']}` p.{c['page']} - {c['score']:.0%} - {flag}")
                        st.caption(c["snippet"])


# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("Upload Documents")
    uploads = st.file_uploader(
        "Choose document files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
    )

    if uploads:
        with st.spinner("Saving and indexing documents..."):
            paths = []
            for f in uploads:
                target = config.DATA_DIR / Path(f.name).name
                target.write_bytes(f.getbuffer())
                paths.append(str(target))
            n_chunks = rag.ingest(paths)
        st.session_state.flash = f"Indexed {len(paths)} file(s), {n_chunks} chunks."
        st.session_state.uploader_key += 1  # resets the uploader
        st.rerun()

    if st.session_state.flash:
        st.success(st.session_state.flash)
        st.session_state.flash = None

    docs = rag.list_documents()
    selected = []

    st.divider()
    st.markdown(f"**Saved documents ({len(docs)})**")
    if docs:
        st.caption("Tick the documents you want to search.")
    for name in docs:
        c1, c2 = st.columns([0.8, 0.2])
        if c1.checkbox(name, value=True, key=f"sel_{name}"):
            selected.append(name)
        if c2.button("✕", key=f"del_{name}", help=f"Delete {name}"):
            rag.delete_document(name)
            st.session_state.pop(f"sel_{name}", None)
            st.rerun()

# ---------------- Header ----------------
left, right = st.columns([0.8, 0.2])
left.title("AI-Powered Document Q&A")
with right:
    st.write("")
    if st.button("New Chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ---------------- Chat ----------------
for msg in st.session_state.history:
    render_message(msg)

if not docs:
    st.info("Upload a PDF or TXT document in the sidebar to get started.")
elif not selected:
    st.warning("Select at least one document in the sidebar.")

question = st.chat_input(
    "Ask a question about your documents...", disabled=not (docs and selected)
)

if question:
    previous = list(st.session_state.history)
    st.session_state.history.append({"role": "user", "content": question})
    render_message(st.session_state.history[-1])

    with st.spinner("Thinking..."):
        try:
            result = rag.ask(question, sources=selected, history=previous)
            reply = {
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
                "model": result["model"],
                "seconds": result.get("seconds"),
                "notes": result.get("notes"),
                "debug": result.get("debug"),
            }
        except RuntimeError as e:
            reply = {"role": "assistant", "content": f"Sorry, something went wrong.\n\n{e}"}

    st.session_state.history.append(reply)
    render_message(reply)
