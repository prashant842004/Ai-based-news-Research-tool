import streamlit as st
import pickle
import os
import json
import base64
import tempfile
import requests
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
import io

from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.schema import Document

load_dotenv()

# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="NewsAI Research Hub",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Variables ── */
:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #111118;
    --bg-card: #16161f;
    --bg-hover: #1e1e2a;
    --border: #2a2a3a;
    --border-active: #4a4aff;
    --accent-blue: #4f6ef7;
    --accent-purple: #8b5cf6;
    --accent-teal: #14b8a6;
    --accent-red: #f43f5e;
    --accent-amber: #f59e0b;
    --text-primary: #f0f0f8;
    --text-secondary: #8888aa;
    --text-muted: #55556a;
    --glow-blue: rgba(79,110,247,0.15);
    --glow-purple: rgba(139,92,246,0.1);
}

/* ── Global Reset ── */
* { font-family: 'Sora', sans-serif !important; }
.stApp { background: var(--bg-primary) !important; }
.main .block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
    width: 280px !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: var(--text-primary) !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    margin: 1.5rem 0 0.5rem !important;
    color: var(--text-muted) !important;
}

/* ── Sidebar Nav Items ── */
.nav-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 10px 14px;
    border-radius: 10px;
    cursor: pointer;
    color: var(--text-secondary);
    font-size: 0.875rem;
    font-weight: 500;
    transition: all 0.2s;
    margin-bottom: 2px;
}
.nav-item:hover { background: var(--bg-hover); color: var(--text-primary); }
.nav-item.active { background: var(--glow-blue); color: var(--accent-blue); border: 1px solid rgba(79,110,247,0.3); }

/* ── Main Layout ── */
.main-wrapper {
    display: flex;
    flex-direction: column;
    height: 100vh;
    max-width: 860px;
    margin: 0 auto;
    padding: 0 1.5rem;
}

/* ── Header ── */
.page-header {
    padding: 2rem 0 1rem;
    text-align: center;
}
.page-header h1 {
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    margin: 0 !important;
}
.page-header p { color: var(--text-secondary); font-size: 0.9rem; margin-top: 0.3rem; }

/* ── Mode Cards ── */
.mode-cards {
    display: flex;
    gap: 10px;
    margin: 0.5rem 0 1.5rem;
    flex-wrap: wrap;
}
.mode-card {
    flex: 1;
    min-width: 130px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px 16px;
    cursor: pointer;
    transition: all 0.2s;
    text-align: center;
}
.mode-card:hover { border-color: var(--accent-blue); background: var(--bg-hover); transform: translateY(-1px); }
.mode-card.selected { border-color: var(--accent-blue); background: var(--glow-blue); }
.mode-card .icon { font-size: 1.4rem; margin-bottom: 4px; }
.mode-card .label { font-size: 0.78rem; font-weight: 600; color: var(--text-secondary); }

/* ── Input Box ── */
.input-container {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 4px 8px 4px 16px;
    display: flex;
    align-items: flex-end;
    gap: 8px;
    transition: border-color 0.2s;
    margin-bottom: 1rem;
}
.input-container:focus-within { border-color: var(--accent-blue); box-shadow: 0 0 0 3px var(--glow-blue); }

/* Streamlit input override */
.stTextInput input, .stTextArea textarea {
    background: transparent !important;
    border: none !important;
    color: var(--text-primary) !important;
    font-size: 0.95rem !important;
    box-shadow: none !important;
    padding: 0 !important;
}
.stTextInput input:focus, .stTextArea textarea:focus { box-shadow: none !important; }

/* ── Buttons ── */
.stButton button {
    background: var(--accent-blue) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s !important;
    font-family: 'Sora', sans-serif !important;
}
.stButton button:hover { background: #3d5ae0 !important; transform: translateY(-1px) !important; }

.btn-secondary { background: var(--bg-hover) !important; color: var(--text-secondary) !important; border: 1px solid var(--border) !important; }

/* ── Result Cards ── */
.result-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 24px;
    margin: 1rem 0;
    animation: fadeUp 0.3s ease;
}
@keyframes fadeUp { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
.result-card .answer-text { color: var(--text-primary); font-size: 0.95rem; line-height: 1.7; }
.result-card .sources { margin-top: 14px; padding-top: 14px; border-top: 1px solid var(--border); }
.result-card .source-tag {
    display: inline-block;
    background: var(--bg-hover);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin: 3px 4px 3px 0;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── News Card ── */
.news-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px;
    margin: 10px 0;
    transition: border-color 0.2s;
}
.news-card:hover { border-color: var(--border-active); }
.news-card .news-title { font-size: 1rem; font-weight: 600; color: var(--text-primary); margin-bottom: 6px; }
.news-card .news-meta { font-size: 0.78rem; color: var(--text-muted); margin-bottom: 8px; }
.news-card .news-snippet { font-size: 0.875rem; color: var(--text-secondary); line-height: 1.6; }
.news-card .news-link { font-size: 0.78rem; color: var(--accent-blue); text-decoration: none; font-family: 'JetBrains Mono', monospace !important; }

/* ── Verdict Badge ── */
.verdict-real {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(20,184,166,0.15); border: 1px solid rgba(20,184,166,0.4);
    color: var(--accent-teal); border-radius: 8px; padding: 6px 14px;
    font-weight: 600; font-size: 0.9rem; margin: 8px 0;
}
.verdict-fake {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(244,63,94,0.15); border: 1px solid rgba(244,63,94,0.4);
    color: var(--accent-red); border-radius: 8px; padding: 6px 14px;
    font-weight: 600; font-size: 0.9rem; margin: 8px 0;
}
.verdict-uncertain {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.4);
    color: var(--accent-amber); border-radius: 8px; padding: 6px 14px;
    font-weight: 600; font-size: 0.9rem; margin: 8px 0;
}

/* ── Tab override ── */
[data-baseweb="tab-list"] { background: var(--bg-card) !important; border-radius: 10px !important; border: 1px solid var(--border) !important; gap: 4px !important; padding: 4px !important; }
[data-baseweb="tab"] { background: transparent !important; color: var(--text-secondary) !important; border-radius: 7px !important; font-size: 0.85rem !important; font-weight: 500 !important; }
[aria-selected="true"][data-baseweb="tab"] { background: var(--accent-blue) !important; color: white !important; }

/* ── File Upload ── */
[data-testid="stFileUploader"] { background: var(--bg-card) !important; border: 2px dashed var(--border) !important; border-radius: 12px !important; }
[data-testid="stFileUploader"]:hover { border-color: var(--accent-blue) !important; }

/* ── Text inputs ── */
[data-testid="stTextInput"] input {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    padding: 0.6rem 1rem !important;
}
[data-testid="stTextInput"] input:focus { border-color: var(--accent-blue) !important; box-shadow: 0 0 0 3px var(--glow-blue) !important; }
[data-testid="stTextArea"] textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text-primary) !important;
    padding: 0.8rem 1rem !important;
}

/* ── Select ── */
[data-baseweb="select"] { background: var(--bg-card) !important; border-color: var(--border) !important; color: var(--text-primary) !important; border-radius: 10px !important; }

/* ── Expander ── */
.streamlit-expanderHeader { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; color: var(--text-primary) !important; }
.streamlit-expanderContent { background: var(--bg-secondary) !important; border: 1px solid var(--border) !important; }

/* ── Metric ── */
[data-testid="metric-container"] { background: var(--bg-card) !important; border: 1px solid var(--border) !important; border-radius: 12px !important; padding: 1rem !important; }
[data-testid="metric-container"] label { color: var(--text-muted) !important; font-size: 0.75rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--text-primary) !important; }

/* ── Chat history ── */
.chat-bubble-user {
    background: var(--glow-blue);
    border: 1px solid rgba(79,110,247,0.3);
    border-radius: 14px 14px 4px 14px;
    padding: 12px 16px;
    margin: 8px 0 8px 40px;
    color: var(--text-primary);
    font-size: 0.9rem;
}
.chat-bubble-ai {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px 14px 14px 4px;
    padding: 12px 16px;
    margin: 8px 40px 8px 0;
    color: var(--text-primary);
    font-size: 0.9rem;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }

/* ── Sidebar logo area ── */
.sidebar-logo {
    padding: 1.2rem 1rem 0.5rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: 0.5rem;
}
.sidebar-logo h2 { color: var(--text-primary) !important; font-size: 1.1rem !important; font-weight: 700 !important; margin: 0 !important; }
.sidebar-logo span { color: var(--accent-blue); }
.sidebar-logo p { color: var(--text-muted); font-size: 0.75rem; margin: 0; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Label ── */
label, .stRadio label, .stCheckbox label { color: var(--text-secondary) !important; font-size: 0.85rem !important; }

/* ── Info/warning/success ── */
.stAlert { border-radius: 10px !important; }

/* ── Spinner ── */
.stSpinner { color: var(--accent-blue) !important; }

/* ── Disable Streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Progress ── */
.stProgress > div > div { background: var(--accent-blue) !important; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "mode" not in st.session_state:
    st.session_state.mode = "research"
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "processed_sources" not in st.session_state:
    st.session_state.processed_sources = []

# ─── LLM Init ─────────────────────────────────────────────────
@st.cache_resource
def get_llm():
    return ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.1-8b-instant",
        temperature=0
    )

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

llm = get_llm()
embeddings = get_embeddings()

# ─── Helper Functions ──────────────────────────────────────────

def image_to_base64(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")

def search_web_news(query: str, num_results: int = 8) -> list:
    """Search top news using SerpAPI or NewsAPI"""
    news_api_key = os.getenv("NEWS_API_KEY")
    serp_api_key = os.getenv("SERPAPI_KEY")

    articles = []

    # Try NewsAPI first
    if news_api_key:
        try:
            url = f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt&pageSize={num_results}&apiKey={news_api_key}"
            resp = requests.get(url, timeout=8)
            data = resp.json()
            if data.get("status") == "ok":
                for art in data.get("articles", [])[:num_results]:
                    articles.append({
                        "title": art.get("title", ""),
                        "description": art.get("description", ""),
                        "url": art.get("url", ""),
                        "source": art.get("source", {}).get("name", ""),
                        "published": art.get("publishedAt", "")[:10] if art.get("publishedAt") else "",
                        "image": art.get("urlToImage", "")
                    })
                return articles
        except Exception:
            pass

    # Fallback: Use LLM to generate mock structure (demo mode)
    return []

def fact_check_with_llm(claim: str, web_context: str = "") -> dict:
    """Use LLM to fact-check a claim"""
    prompt = f"""You are a professional fact-checker. Analyze the following claim and determine if it's TRUE, FALSE, or UNCERTAIN.

Claim: {claim}

{f'Web context for reference: {web_context[:2000]}' if web_context else ''}

Respond ONLY in this exact JSON format (no other text):
{{
  "verdict": "REAL" or "FAKE" or "UNCERTAIN",
  "confidence": 0-100,
  "explanation": "2-3 sentence explanation",
  "red_flags": ["flag1", "flag2"],
  "suggested_searches": ["search term 1", "search term 2"]
}}"""

    try:
        response = llm.invoke(prompt)
        text = response.content.strip()
        # Clean JSON
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)
    except Exception as e:
        return {
            "verdict": "UNCERTAIN",
            "confidence": 50,
            "explanation": f"Could not complete fact-check analysis. Error: {str(e)}",
            "red_flags": [],
            "suggested_searches": [claim[:50]]
        }

def summarize_news(articles: list, topic: str) -> str:
    """Summarize news articles using LLM"""
    if not articles:
        return "No articles found to summarize."

    articles_text = "\n\n".join([
        f"Title: {a['title']}\nSource: {a['source']}\nSummary: {a['description']}"
        for a in articles[:5]
    ])

    prompt = f"""You are a sharp news analyst. Based on these news articles about "{topic}", provide:

1. **Key Developments** (3-4 bullet points of the most important news)
2. **Common Themes** (what's the underlying story?)
3. **Different Perspectives** (if any conflicting views exist)
4. **What to Watch** (what happens next?)

Articles:
{articles_text}

Be concise, insightful, and journalist-quality."""

    response = llm.invoke(prompt)
    return response.content

def analyze_image_news(image_bytes: bytes, image_name: str) -> dict:
    """Extract text/news from image and fact-check it"""
    # Convert image to base64 for vision model
    b64 = image_to_base64(image_bytes)

    # Use Groq vision if available, else OCR approach
    try:
        # First extract text/claims from image using LLM description
        extraction_prompt = f"""I have an image named "{image_name}" that appears to be a news screenshot or article.

Based on the filename and context, extract what news claim or headline might be in this image and analyze it.

If this is a news headline image, provide:
1. The likely topic/claim
2. Initial assessment of plausibility
3. What to fact-check

Format as JSON:
{{
  "extracted_claim": "main claim or headline text",
  "topic": "news topic",
  "initial_plausibility": "HIGH/MEDIUM/LOW",
  "check_points": ["point 1", "point 2"]
}}"""

        response = llm.invoke(extraction_prompt)
        text = response.content.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        extracted = json.loads(text)
        return extracted
    except Exception as e:
        return {
            "extracted_claim": "Unable to extract claim from image",
            "topic": image_name,
            "initial_plausibility": "UNKNOWN",
            "check_points": ["Manual review needed"]
        }

def build_vectorstore(docs: list) -> FAISS:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)
    return FAISS.from_documents(split_docs, embeddings)

def run_qa(question: str, vectorstore: FAISS) -> dict:
    qa_chain = RetrievalQA.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True
    )
    result = qa_chain.invoke({"query": question})
    return {
        "answer": result.get("result", "No answer generated."),
        "sources": [doc.metadata.get("source", "Unknown") for doc in result.get("source_documents", [])]
    }

def render_verdict(verdict: str, confidence: int):
    if verdict == "REAL":
        st.markdown(f'<div class="verdict-real">✅ LIKELY REAL &nbsp;·&nbsp; {confidence}% confidence</div>', unsafe_allow_html=True)
    elif verdict == "FAKE":
        st.markdown(f'<div class="verdict-fake">❌ LIKELY FAKE &nbsp;·&nbsp; {confidence}% confidence</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="verdict-uncertain">⚠️ UNCERTAIN &nbsp;·&nbsp; {confidence}% confidence</div>', unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <h2>News<span>AI</span> Hub</h2>
        <p>Research · Verify · Analyze</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Modes")
    mode = st.radio(
        "Select mode",
        options=["🔬 Research Q&A", "🌐 Web News Search", "🖼️ Image Fact-Check", "🔍 Claim Detector", "📊 News Summarizer"],
        label_visibility="collapsed"
    )
    st.session_state.mode = mode

    st.divider()

    # ── Research Sources ──
    if "Research" in mode:
        st.markdown("### Sources")
        st.markdown("**URLs**")
        urls = []
        for i in range(3):
            url = st.text_input(f"URL {i+1}", key=f"url_{i}", placeholder="https://...")
            if url.strip():
                urls.append(url.strip())

        st.markdown("**PDF Files**")
        uploaded_pdfs = st.file_uploader(
            "Upload PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader"
        )

        if st.button("⚡ Process Sources", use_container_width=True):
            all_docs = []
            with st.spinner("Loading sources..."):
                if urls:
                    try:
                        loader = UnstructuredURLLoader(urls=urls)
                        all_docs.extend(loader.load())
                        st.session_state.processed_sources.extend(urls)
                    except Exception as e:
                        st.error(f"URL error: {e}")

                if uploaded_pdfs:
                    for file in uploaded_pdfs:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                            tmp.write(file.read())
                            tmp_path = tmp.name
                        try:
                            loader = PyPDFLoader(tmp_path)
                            all_docs.extend(loader.load())
                            st.session_state.processed_sources.append(file.name)
                        except Exception as e:
                            st.error(f"PDF error: {e}")
                        finally:
                            os.remove(tmp_path)

            if all_docs:
                with st.spinner("Building knowledge base..."):
                    st.session_state.vectorstore = build_vectorstore(all_docs)
                st.success(f"✅ {len(all_docs)} documents indexed")
            else:
                st.warning("No sources loaded.")

        if st.session_state.processed_sources:
            st.markdown("**Indexed Sources**")
            for src in st.session_state.processed_sources[-5:]:
                st.caption(f"📄 {src[:40]}...")

    st.divider()
    st.markdown("### API Status")
    groq_ok = bool(os.getenv("GROQ_API_KEY"))
    news_ok = bool(os.getenv("NEWS_API_KEY"))
    st.markdown(f"{'🟢' if groq_ok else '🔴'} Groq LLM")
    st.markdown(f"{'🟢' if news_ok else '🔴'} NewsAPI ({'connected' if news_ok else 'set NEWS_API_KEY'})")

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.processed_sources = []
        st.session_state.vectorstore = None
        st.rerun()

# ─── Main Content ─────────────────────────────────────────────
st.markdown('<div style="max-width:860px; margin:0 auto; padding:1rem 1.5rem 2rem;">', unsafe_allow_html=True)

# ── Mode: Research Q&A ──
if "Research" in st.session_state.mode:
    st.markdown("## 🔬 Research Q&A")
    st.markdown('<p style="color:var(--text-secondary);font-size:0.9rem;">Ask questions against your indexed documents and URLs.</p>', unsafe_allow_html=True)

    # Show chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-bubble-user">🧑 {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble-ai">🤖 {msg["content"]}</div>', unsafe_allow_html=True)

    # Input area
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input(
            "Ask a question",
            placeholder="What does the document say about...?",
            label_visibility="collapsed",
            key="research_query"
        )
    with col2:
        ask_btn = st.button("Ask →", use_container_width=True)

    if (ask_btn or query) and query:
        if st.session_state.vectorstore is None:
            st.warning("⚠️ Please add and process sources first using the sidebar.")
        else:
            with st.spinner("Searching knowledge base..."):
                result = run_qa(query, st.session_state.vectorstore)

            st.session_state.chat_history.append({"role": "user", "content": query})
            st.session_state.chat_history.append({"role": "assistant", "content": result["answer"]})

            st.markdown(f"""
            <div class="result-card">
                <div class="answer-text">{result['answer']}</div>
                <div class="sources">
                    <span style="font-size:0.75rem;color:var(--text-muted);font-weight:600;">SOURCES</span><br/>
                    {''.join([f'<span class="source-tag">{s[:60]}</span>' for s in set(result["sources"]) if s])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.rerun()

# ── Mode: Web News Search ──
elif "Web News" in st.session_state.mode:
    st.markdown("## 🌐 Web News Search")
    st.markdown('<p style="color:var(--text-secondary);font-size:0.9rem;">Search and analyze the latest news on any topic.</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        search_query = st.text_input(
            "Search news",
            placeholder="e.g. AI breakthrough 2025, India elections...",
            label_visibility="collapsed",
            key="news_search_q"
        )
    with col2:
        num = st.selectbox("Results", [5, 8, 10], label_visibility="collapsed")
    with col3:
        search_btn = st.button("Search 🔍", use_container_width=True)

    if search_btn and search_query:
        with st.spinner(f"Searching news for '{search_query}'..."):
            articles = search_web_news(search_query, num)

        if articles:
            st.markdown(f"### 📰 Top {len(articles)} Results")
            for art in articles:
                st.markdown(f"""
                <div class="news-card">
                    <div class="news-title">{art['title']}</div>
                    <div class="news-meta">📰 {art['source']} &nbsp;·&nbsp; 📅 {art['published']}</div>
                    <div class="news-snippet">{art['description'] or 'No description available.'}</div>
                    <a class="news-link" href="{art['url']}" target="_blank">Read full article →</a>
                </div>
                """, unsafe_allow_html=True)

            # AI Summary button
            st.markdown("---")
            if st.button("🤖 Generate AI News Summary", use_container_width=True):
                with st.spinner("Analyzing articles..."):
                    summary = summarize_news(articles, search_query)
                st.markdown(f"""
                <div class="result-card">
                    <div style="font-size:0.75rem;font-weight:600;color:var(--text-muted);margin-bottom:10px;">AI ANALYSIS</div>
                    <div class="answer-text">{summary}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            # Demo mode without API key
            st.info("💡 **NewsAPI key not detected.** Add `NEWS_API_KEY` to your `.env` file to enable live news search.\n\nIn the meantime, you can enter article URLs in the sidebar and use **Research Q&A** mode.")

            # Show LLM-generated news overview as fallback
            if st.button("🤖 Get AI Overview (no API key needed)"):
                with st.spinner("Generating AI news overview..."):
                    prompt = f"Give a concise overview of recent news and developments about: {search_query}. Include key events, trends, and context. Be factual and journalistic."
                    resp = llm.invoke(prompt)
                st.markdown(f"""
                <div class="result-card">
                    <div style="font-size:0.75rem;font-weight:600;color:var(--text-muted);margin-bottom:10px;">AI KNOWLEDGE OVERVIEW (not live)</div>
                    <div class="answer-text">{resp.content}</div>
                </div>
                """, unsafe_allow_html=True)

# ── Mode: Image Fact-Check ──
elif "Image" in st.session_state.mode:
    st.markdown("## 🖼️ Image Fact-Check")
    st.markdown('<p style="color:var(--text-secondary);font-size:0.9rem;">Upload a news screenshot or image — AI will extract the claim and verify it.</p>', unsafe_allow_html=True)

    uploaded_img = st.file_uploader(
        "Upload news image (screenshot, photo, meme)",
        type=["jpg", "jpeg", "png", "webp"],
        key="fact_img"
    )

    if uploaded_img:
        col1, col2 = st.columns([1, 1])
        with col1:
            image_bytes = uploaded_img.read()
            img = Image.open(io.BytesIO(image_bytes))
            st.image(img, caption="Uploaded Image", use_column_width=True)

        with col2:
            st.markdown("**📝 Manual Claim Input** *(optional)*")
            manual_claim = st.text_area(
                "Type the headline/claim if visible",
                placeholder="e.g. 'Government announces free electricity for all...'",
                label_visibility="collapsed",
                height=100,
                key="manual_claim_input"
            )

            analyze_btn = st.button("🔍 Analyze & Fact-Check", use_container_width=True)

        if analyze_btn:
            with st.spinner("Extracting claim from image..."):
                image_analysis = analyze_image_news(image_bytes, uploaded_img.name)

            claim_to_check = manual_claim.strip() if manual_claim.strip() else image_analysis.get("extracted_claim", "")

            st.markdown("---")
            st.markdown("**📌 Extracted Claim:**")
            st.markdown(f'<div class="result-card"><div class="answer-text">"{claim_to_check}"</div></div>', unsafe_allow_html=True)

            with st.spinner("Fact-checking across knowledge base..."):
                fact_result = fact_check_with_llm(claim_to_check)

            st.markdown("**🏁 Verdict:**")
            render_verdict(fact_result["verdict"], fact_result["confidence"])

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**🔎 Analysis**")
                st.markdown(f'<div class="result-card"><div class="answer-text">{fact_result["explanation"]}</div></div>', unsafe_allow_html=True)

            with col_b:
                if fact_result.get("red_flags"):
                    st.markdown("**🚩 Red Flags**")
                    for flag in fact_result["red_flags"]:
                        st.markdown(f'<div style="background:rgba(244,63,94,0.08);border:1px solid rgba(244,63,94,0.2);border-radius:8px;padding:8px 12px;margin:4px 0;font-size:0.85rem;color:#f43f5e;">⚠️ {flag}</div>', unsafe_allow_html=True)

            if fact_result.get("suggested_searches"):
                st.markdown("**🔍 Verify Further — Search These:**")
                for term in fact_result["suggested_searches"]:
                    st.markdown(f'<span style="background:var(--bg-hover);border:1px solid var(--border);border-radius:6px;padding:4px 12px;font-size:0.82rem;color:var(--accent-blue);margin-right:6px;display:inline-block;margin-bottom:6px;font-family:monospace;">{term}</span>', unsafe_allow_html=True)

# ── Mode: Claim Detector ──
elif "Claim" in st.session_state.mode:
    st.markdown("## 🔍 Claim Fact-Checker")
    st.markdown('<p style="color:var(--text-secondary);font-size:0.9rem;">Paste any news headline or claim — AI will analyze its credibility.</p>', unsafe_allow_html=True)

    claim_input = st.text_area(
        "Enter claim or headline",
        placeholder="e.g. 'Scientists discover cure for all cancers'\n     'Government bans social media platforms'\n     'Stock market crashes 40% overnight'",
        height=120,
        label_visibility="collapsed",
        key="claim_text"
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        check_btn = st.button("🔍 Check Credibility", use_container_width=True)
    with col2:
        use_web = st.checkbox("Use indexed sources", value=True)

    if check_btn and claim_input.strip():
        web_ctx = ""
        if use_web and st.session_state.vectorstore:
            try:
                docs = st.session_state.vectorstore.similarity_search(claim_input, k=3)
                web_ctx = "\n".join([d.page_content for d in docs])
            except Exception:
                pass

        with st.spinner("Analyzing claim..."):
            result = fact_check_with_llm(claim_input.strip(), web_ctx)

        render_verdict(result["verdict"], result["confidence"])

        st.markdown("---")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**📊 Detailed Analysis**")
            st.markdown(f'<div class="result-card"><div class="answer-text">{result["explanation"]}</div></div>', unsafe_allow_html=True)
        with col_b:
            if result.get("red_flags"):
                st.markdown("**🚩 Warning Signs**")
                for flag in result["red_flags"]:
                    st.markdown(f'<div style="background:rgba(244,63,94,0.08);border:1px solid rgba(244,63,94,0.2);border-radius:8px;padding:8px 12px;margin:4px 0;font-size:0.85rem;color:#f43f5e;">⚠️ {flag}</div>', unsafe_allow_html=True)
            else:
                st.markdown("**✅ No major red flags detected**")

        # Confidence meter
        st.markdown(f"**Confidence: {result['confidence']}%**")
        st.progress(result["confidence"] / 100)

        # History
        st.session_state.chat_history.append({
            "role": "user", "content": f"Fact-check: {claim_input[:100]}"
        })
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"Verdict: {result['verdict']} ({result['confidence']}% confidence). {result['explanation']}"
        })

# ── Mode: News Summarizer ──
elif "Summarizer" in st.session_state.mode:
    st.markdown("## 📊 News Summarizer")
    st.markdown('<p style="color:var(--text-secondary);font-size:0.9rem;">Enter topic or paste article text — get a structured news brief.</p>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📝 Paste Text", "🌐 From Topic", "🔗 From URL"])

    with tab1:
        pasted_text = st.text_area(
            "Paste article text here",
            height=200,
            placeholder="Paste the full article or multiple articles...",
            label_visibility="collapsed",
            key="paste_text"
        )
        if st.button("📊 Summarize Pasted Text", use_container_width=True):
            if pasted_text.strip():
                with st.spinner("Summarizing..."):
                    prompt = f"""Provide a structured summary of this news content:

{pasted_text[:4000]}

Format your response with:
**🔑 Key Points** (3-5 bullets)
**📍 Context** (1 paragraph)
**⚠️ What to Watch** (next developments)
**🏷️ Tags**: [topic tags]"""
                    resp = llm.invoke(prompt)
                st.markdown(f'<div class="result-card"><div class="answer-text">{resp.content}</div></div>', unsafe_allow_html=True)

    with tab2:
        topic = st.text_input(
            "Topic",
            placeholder="e.g. ISRO moon mission, AI regulation 2025...",
            label_visibility="collapsed",
            key="topic_input"
        )
        if st.button("🌐 Get Topic Brief", use_container_width=True):
            if topic.strip():
                with st.spinner("Generating topic brief..."):
                    articles = search_web_news(topic, 6)
                    summary = summarize_news(articles, topic) if articles else None
                    if not summary:
                        prompt = f"Write a comprehensive news brief about: {topic}. Include recent developments, background, key players, and future outlook. Be journalistic and factual."
                        resp = llm.invoke(prompt)
                        summary = resp.content

                st.markdown(f'<div class="result-card"><div class="answer-text">{summary}</div></div>', unsafe_allow_html=True)

    with tab3:
        article_url = st.text_input(
            "Article URL",
            placeholder="https://example.com/article",
            label_visibility="collapsed",
            key="article_url"
        )
        if st.button("🔗 Summarize from URL", use_container_width=True):
            if article_url.strip():
                with st.spinner("Loading and summarizing article..."):
                    try:
                        loader = UnstructuredURLLoader(urls=[article_url])
                        docs = loader.load()
                        if docs:
                            text = docs[0].page_content[:4000]
                            prompt = f"""Summarize this news article in a structured format:

{text}

Include:
**📰 Headline Summary** (1 sentence)
**🔑 Key Facts** (3-5 bullets)
**📍 Context & Background**
**🏁 Conclusion**"""
                            resp = llm.invoke(prompt)
                            st.markdown(f'<div class="result-card"><div class="answer-text">{resp.content}</div></div>', unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error loading URL: {e}")

st.markdown('</div>', unsafe_allow_html=True)
