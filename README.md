<div align="center">

# 🔍 NewsAI Research Hub

### *An Intelligent News Research & Fact-Verification Platform*

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://langchain.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA%203.1-F54E00?style=for-the-badge&logo=lightning&logoColor=white)](https://groq.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector%20DB-0064A5?style=for-the-badge&logo=meta&logoColor=white)](https://faiss.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)](LICENSE)

<br/>

> **Ask questions about any news article. Detect fake news in seconds. Verify claims with AI. Search & summarize the latest headlines — all in one place.**

<br/>

![NewsAI Demo Banner](https://placehold.co/900x400/0a0a0f/4f6ef7?text=NewsAI+Research+Hub&font=montserrat)

</div>

---

## ✨ What Makes This Special

Most news tools just display articles. **NewsAI Research Hub goes deeper** — it reads, understands, cross-references, and challenges the news for you.

Built for journalists, researchers, students, and anyone who refuses to believe everything they read.

---

## 🚀 Five Intelligent Modes

### 🔬 1. Research Q&A
Feed it URLs and PDFs. Ask it anything. Get sourced, referenced answers.

- Processes multiple news URLs and PDF documents simultaneously
- Splits, embeds, and indexes content into a **FAISS vector store**
- Semantic retrieval surfaces the most relevant chunks to answer your question
- Every answer comes with **cited sources** — no hallucinations unchecked
- Persistent chat history with a clean bubble UI (user / AI alternating)

### 🌐 2. Web News Search
Real-time news discovery powered by **NewsAPI** with AI-generated analysis.

- Search any topic and get the top 8–10 latest articles instantly
- Each result shows source, publication date, snippet, and direct link
- Hit **"AI Summary"** to get a structured analyst-grade brief across all results
- Graceful fallback to LLM knowledge overview if API key is not configured

### 🖼️ 3. Image Fact-Check
*"Is this WhatsApp forward real?"* — Upload and find out.

- Upload screenshots, photos, news clippings, or memes
- AI extracts the core claim from the image automatically
- Optional manual claim input for precision
- Full **REAL / FAKE / UNCERTAIN** verdict with confidence score
- Highlights specific **red flags** (sensationalist language, missing sources, implausible claims)
- Suggests follow-up search terms to verify independently

### 🔍 4. Claim Detector
Paste any headline. Get a credibility verdict in under 5 seconds.

- Accepts raw text claims or full headlines
- Cross-references against your indexed knowledge base if available
- Returns **confidence percentage** with a visual progress bar
- Lists specific warning signs detected in the claim
- Logs all checks to conversation history for review

### 📊 5. News Summarizer
Three ways to get a sharp, structured news brief:

| Sub-mode | How it works |
|---|---|
| **Paste Text** | Paste a full article → get key points, context, and conclusion |
| **From Topic** | Enter a keyword → AI generates a topic brief with trends & outlook |
| **From URL** | Drop a link → app loads and summarizes the article directly |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     NewsAI Research Hub                      │
│                      (Streamlit UI)                          │
└───────────────────────────┬─────────────────────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         │                  │                  │
    ┌────▼─────┐     ┌──────▼──────┐    ┌──────▼──────┐
    │  Ingestion│     │  Retrieval  │    │  Generation │
    │  Pipeline │     │   Engine    │    │   Engine    │
    └────┬─────┘     └──────┬──────┘    └──────┬──────┘
         │                  │                  │
  URLs / PDFs /      FAISS Vector DB     Groq (LLaMA 3.1)
  Images / Text    + HuggingFace          Fast Inference
                    Embeddings            < 1s latency
                   (MiniLM-L6-v2)
```

**Data Flow:**
1. **Ingest** — URLs/PDFs loaded → cleaned → chunked (1000 tokens, 100 overlap)
2. **Embed** — Chunks encoded via `sentence-transformers/all-MiniLM-L6-v2`
3. **Index** — Vectors stored in in-memory FAISS for millisecond similarity search
4. **Retrieve** — Top-k chunks fetched per query using cosine similarity
5. **Generate** — LLaMA 3.1 8B (via Groq) synthesizes the final answer with context

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Frontend** | Streamlit 1.32+ | UI framework with custom CSS/dark theme |
| **LLM** | Groq + LLaMA 3.1 8B Instant | Fast inference (~400 tok/s) |
| **Orchestration** | LangChain 0.2 | Chains, retrieval, document loading |
| **Vector DB** | FAISS (Meta) | Semantic similarity search |
| **Embeddings** | HuggingFace MiniLM-L6-v2 | Text vectorization (384-dim) |
| **News API** | NewsAPI.org | Live article search |
| **Document Loaders** | UnstructuredURLLoader, PyPDFLoader | Multi-source ingestion |
| **Image Processing** | Pillow | Image upload & preview |
| **Env Management** | python-dotenv | Secure API key handling |

---

## ⚡ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/newsai-research-hub.git
cd newsai-research-hub
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API keys
```bash
cp .env.example .env
```

Edit `.env`:
```env
GROQ_API_KEY=your_groq_api_key_here        # Required — groq.com (free)
NEWS_API_KEY=your_newsapi_key_here          # Optional — newsapi.org (free tier)
```

### 5. Launch
```bash
streamlit run app.py
```

Open **http://localhost:8501** 🎉

---

## 🔑 API Keys Setup

| Key | Required | Where to get | Free Tier |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | [console.groq.com](https://console.groq.com) | ✅ Free |
| `NEWS_API_KEY` | ⚡ Recommended | [newsapi.org/register](https://newsapi.org/register) | ✅ 100 req/day |

> Without `NEWS_API_KEY`, the Web Search mode falls back to LLM-based overviews. All other modes work fully with just the Groq key.

---

## 📁 Project Structure

```
newsai-research-hub/
│
├── app.py                  # Main application — all 5 modes + UI
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
├── .env                    # Your keys (gitignored)
├── README.md               # This file
│
└── .gitignore
```

---

## 🎨 UI Design Highlights

The interface is built from scratch with **custom CSS** injected into Streamlit:

- **Dark theme** (`#0a0a0f` base) with layered depth — no eye strain for long sessions
- **Sora + JetBrains Mono** font pairing — clean, modern, technical
- **Glowing focus states** on inputs — blue accent with subtle box-shadow
- **Animated result cards** — `fadeUp` keyframe on every new answer
- **Color-coded verdicts** — teal (real), red (fake), amber (uncertain)
- **Chat bubble layout** — alternating user/AI bubbles with distinct shapes
- Fully responsive sidebar with API status indicators

---

## 🗺️ Feature Roadmap

- [ ] **Multilingual support** — fact-check news in Hindi, Spanish, French
- [ ] **Browser extension** — highlight any text on the web, fact-check instantly
- [ ] **Timeline view** — track how a story evolved across dates
- [ ] **Source credibility scores** — bias ratings for news outlets
- [ ] **Export reports** — PDF export of research sessions
- [ ] **Real-time alerts** — monitor topics and get notified of new developments
- [ ] **MongoDB persistence** — save and revisit past research sessions
- [ ] **Multi-user support** — team collaboration on shared knowledge bases

---

## 🧠 Key Engineering Decisions

**Why FAISS over Chroma/Pinecone?**
FAISS runs fully in-memory with zero infrastructure — perfect for a portable research tool. For production scale, a switch to Pinecone would be a one-line change in LangChain.

**Why Groq over OpenAI?**
Groq's LPU hardware delivers ~400 tokens/second — answers feel instant. LLaMA 3.1 8B performs comparably to GPT-3.5 on factual QA tasks at zero cost on the free tier.

**Why `sentence-transformers/all-MiniLM-L6-v2`?**
It's 80MB, runs on CPU in under 100ms per chunk, and scores 85%+ on semantic textual similarity benchmarks — the sweet spot of speed vs. quality for a local embedding model.

---

## 🤝 Contributing

Contributions are welcome. Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add: your feature description'`)
4. Push and open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with curiosity, LangChain, and too many news tabs.**

*If this saved you from sharing fake news — give it a ⭐*

[![GitHub stars](https://img.shields.io/github/stars/yourusername/newsai-research-hub?style=social)](https://github.com/yourusername/newsai-research-hub)

</div>
