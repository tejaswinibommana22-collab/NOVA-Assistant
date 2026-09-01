"""
NOVA — Universal Knowledge Assistant
A real Python project combining:
  - Machine Learning: sentence-transformer embeddings + FAISS similarity search
  - AI: Google Gemini for grounded answer generation
  - App: Streamlit chat interface with document upload

Run with:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import faiss
import re
import glob
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# ── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="NOVA — Knowledge Assistant", page_icon="🟠", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

    html, body, .stApp, [data-testid="stAppViewContainer"] {
        background-color: #0B1120 !important;
        background-image:
            radial-gradient(ellipse 900px 500px at 50% 50%, rgba(232,163,61,0.10), transparent),
            radial-gradient(ellipse 700px 500px at 50% 50%, rgba(70,100,200,0.09), transparent) !important;
        background-size: 200% 200%, 200% 200% !important;
        background-position: 5% 0%, 100% 100% !important;
        animation: nova-bg-shift 18s ease-in-out infinite !important;
        font-family: 'Inter', sans-serif;
    }
    @keyframes nova-bg-shift {
        0%   { background-position: 5% 0%,   100% 100%; }
        50%  { background-position: 35% 30%, 65% 70%;   }
        100% { background-position: 5% 0%,   100% 100%; }
    }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #F3F5FA; }

    /* NOVA header block */
    .nova-header-wrap { display: flex; align-items: center; gap: 16px; margin-bottom: 4px; }
    .nova-orb {
        width: 46px; height: 46px; border-radius: 50%; flex-shrink: 0;
        background: radial-gradient(circle at 35% 30%, #FFD98E, #E8A33D 55%, #A5641A 100%);
        box-shadow: 0 0 22px rgba(232,163,61,0.6), inset 0 0 10px rgba(255,255,255,0.25);
        animation: nova-breathe 3s ease-in-out infinite;
        position: relative;
    }
    .nova-orb::after {
        content: ''; position: absolute; inset: -7px; border-radius: 50%;
        border: 1px solid rgba(232,163,61,0.35); animation: nova-ring 3s ease-in-out infinite;
    }
    @keyframes nova-breathe {
        0%, 100% { box-shadow: 0 0 14px rgba(232,163,61,0.45), inset 0 0 8px rgba(255,255,255,0.2); }
        50% { box-shadow: 0 0 28px rgba(232,163,61,0.8), inset 0 0 12px rgba(255,255,255,0.3); }
    }
    @keyframes nova-ring {
        0%, 100% { transform: scale(1); opacity: 0.6; }
        50% { transform: scale(1.15); opacity: 0.1; }
    }
    .nova-title { font-size: 30px; font-weight: 700; letter-spacing: 0.5px; color: #F3F5FA; line-height: 1; }
    .nova-status-line {
        display: flex; align-items: center; gap: 7px; margin-top: 8px;
        font-size: 11.5px; color: #4ADE80; font-weight: 500; letter-spacing: 0.3px;
    }
    .nova-status-dot { width: 6px; height: 6px; border-radius: 50%; background: #4ADE80; box-shadow: 0 0 7px #4ADE80; animation: nova-blink 1.8s infinite; }
    @keyframes nova-blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
    .nova-scanline {
        height: 1px; width: 100%; margin: 18px 0 22px;
        background: linear-gradient(90deg, transparent, rgba(232,163,61,0.5), transparent);
    }

    /* Chat bubbles */
    .stChatMessage {
        background-color: #1A2238 !important;
        border: 1px solid #232C45;
        border-radius: 12px !important;
    }
    .source-tag {
        display: inline-block; margin-top: 8px; font-size: 11px;
        color: #E8A33D; background: #382B16; padding: 3px 9px; border-radius: 5px;
        letter-spacing: 0.2px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #10182A; border-right: 1px solid #232C45; }

    /* Chat input */
    .stChatInput textarea, div[data-testid="stChatInput"] textarea {
        background-color: #1A2238 !important; border: 1px solid #232C45 !important; color: #F3F5FA !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="nova-header-wrap">
    <div class="nova-orb"></div>
    <div>
        <div class="nova-title">NOVA</div>
    </div>
</div>
<div class="nova-status-line"><div class="nova-status-dot"></div>SYSTEM ONLINE · GROUNDED IN LOADED DOCUMENTS</div>
<div class="nova-scanline"></div>
""", unsafe_allow_html=True)

st.caption("Ask questions grounded in any documents — built with Python, ML (embeddings + FAISS), and Claude AI")

# ── Chunking helper (Python) ────────────────────────────────────────────────
def chunk_text(text, source_name):
    # Split on markdown headers if present, otherwise on paragraphs
    if "## " in text:
        sections = re.split(r'\n(?=## )', text)
    else:
        sections = re.split(r'\n\s*\n', text)
    chunks = []
    for sec in sections:
        sec = sec.strip()
        if len(sec) < 20:
            continue
        chunks.append({"text": sec, "source": source_name})
    return chunks

# ── Load default TechNova docs (used if nothing is uploaded) ───────────────
@st.cache_resource
def load_default_chunks():
    all_chunks = []
    for filepath in glob.glob("company_docs/*.md"):
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        all_chunks.extend(chunk_text(content, filepath.split("/")[-1]))
    return all_chunks

# ── Build embeddings + FAISS index (Machine Learning) ───────────────────────
@st.cache_resource
def get_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

def build_index(chunks, embedder):
    texts = [c["text"] for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

# ── Retrieval (ML) ───────────────────────────────────────────────────────────
def retrieve(query, embedder, index, chunks, top_k=3):
    query_vec = embedder.encode([query]).astype("float32")
    _, indices = index.search(query_vec, top_k)
    return [chunks[i] for i in indices[0]]

# ── Generate grounded answer (AI) ────────────────────────────────────────────
def ask_nova(question, embedder, index, chunks, api_key, assistant_name="NOVA"):
    context_chunks = retrieve(question, embedder, index, chunks)
    context_text = "\n\n".join([f"[Source: {c['source']}]\n{c['text']}" for c in context_chunks])
    sources = sorted(set(c["source"] for c in context_chunks))

    system_prompt = (
        f"You are {assistant_name}, a knowledge assistant. "
        "Answer the user's question using ONLY the provided context. "
        "If the answer isn't in the context, say you don't have that information. "
        "Keep answers short (2-4 sentences), clear, and friendly."
    )
    full_prompt = f"{system_prompt}\n\nContext:\n{context_text}\n\nQuestion: {question}"

    genai.configure(api_key=api_key)

    # Try a list of model names in order, in case one is deprecated/renamed
    candidate_models = [
        "gemini-3.6-flash",
        "gemini-flash-latest",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-pro-latest",
    ]
    last_error = None
    for model_name in candidate_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(full_prompt)
            return response.text, sources
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"All model attempts failed. Last error: {last_error}")

# ── Sidebar: upload + settings ──────────────────────────────────────────────
embedder = get_embedder()

with st.sidebar:
    st.subheader("📚 Knowledge base")
    uploaded_files = st.file_uploader(
        "Upload your own documents (.txt or .md)",
        type=["txt", "md"],
        accept_multiple_files=True,
        help="Upload any documents — policies, notes, manuals. NOVA will answer questions grounded in them.",
    )

    if uploaded_files:
        active_chunks = []
        for f in uploaded_files:
            content = f.read().decode("utf-8", errors="ignore")
            active_chunks.extend(chunk_text(content, f.name))
        st.success(f"Using {len(uploaded_files)} uploaded document(s)")
        source_label = [f.name for f in uploaded_files]
    else:
        active_chunks = load_default_chunks()
        st.info("Using default TechNova sample documents")
        source_label = [f.split("/")[-1] for f in glob.glob("company_docs/*.md")]

    for s in source_label:
        st.markdown(f"📄 {s}")

    st.divider()
    api_key = st.text_input("Google Gemini API key", type="password", help="Get one FREE at aistudio.google.com/apikey")
    st.divider()

    st.caption("Try asking:")
    default_questions = [
        "How many sick leaves do I get?",
        "My laptop screen is broken, what do I do?",
        "What happens on my first day?",
        "What's the notice period if I resign?",
    ]
    for q in default_questions:
        st.button(q, use_container_width=True, key=q)

# Rebuild index whenever the active document set changes
docs_signature = tuple(sorted(source_label))
if "docs_signature" not in st.session_state or st.session_state.docs_signature != docs_signature:
    st.session_state.index = build_index(active_chunks, embedder)
    st.session_state.chunks = active_chunks
    st.session_state.docs_signature = docs_signature
    st.session_state.messages = [
        {"role": "assistant", "content": f"Systems online. I've indexed {len(source_label)} document(s) — ask me anything.", "source": None}
    ]

index = st.session_state.index
chunks = st.session_state.chunks

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("source"):
            st.markdown(f"<span class='source-tag'>📄 {', '.join(msg['source'])}</span>", unsafe_allow_html=True)

clicked_question = None
for q in default_questions:
    if st.session_state.get(q):
        clicked_question = q

prompt = st.chat_input("Ask a question about the loaded documents...") or clicked_question

if prompt:
    if not api_key:
        st.error("Please enter your Google Gemini API key in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt, "source": None})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                answer, sources = ask_nova(prompt, embedder, index, chunks, api_key)
                st.write(answer)
                st.markdown(f"<span class='source-tag'>📄 {', '.join(sources)}</span>", unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": answer, "source": sources})
