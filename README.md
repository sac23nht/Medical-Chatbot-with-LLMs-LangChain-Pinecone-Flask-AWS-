# 🩺 Medical Chatbot: RAG Q&A with sources

A **retrieval-augmented generation (RAG)** chatbot built with **Streamlit**. Ask a medical
question and it answers from a real reference book, the *Gale Encyclopedia of Medicine*
(2nd ed., 2002), and **shows the exact passages it used as sources**.

> Educational demo, **not medical advice**. Answers are AI-generated from a 2002 book and may be
> incomplete or out of date.

**Developed by Saijaya Rami Reddy Chilekampalli.**

## What you see

A two-panel page: the **chat on the left** (with example questions and a 📚 Sources panel under
every answer) and the **project details on the right**: what the app is, the source book that was
embedded (`data/Medical_book.pdf`), a simple workflow diagram (one-time setup and per-question
flow), the tools used, and the developer. On a phone the panels stack.

## How it works

```
Your question -> Embed (MiniLM-L6-v2) -> Retrieve top 3 (Pinecone) -> Generate (Llama 3.1 8B) -> Answer + sources
```

1. The question is turned into a 384-number vector by the `all-MiniLM-L6-v2` embedding model.
2. Pinecone returns the 3 most similar passages out of ~5,900 chunks of the book.
3. The language model is told to answer **only** from those passages, or say it doesn't know.
4. The app shows the answer plus a **📚 Sources** panel with the retrieved text.

Setup, done once: the book's PDF is split into ~500-character chunks, embedded, and stored in a
Pinecone index (`src/store_index.py`).

## Tech stack

| Area | Tools |
|---|---|
| AI & retrieval | RAG, LangChain, Pinecone, Sentence-Transformers (MiniLM-L6-v2), Llama 3.1 8B, Hugging Face Inference API |
| App | Python, Streamlit |
| Data pipeline | PyPDF, text chunking |
| Deployment | Docker, Streamlit Community Cloud / Render, Git & GitHub |

The embeddings and the language model are **hosted APIs**, so the app needs no GPU and runs in
roughly 150 MB of RAM, which fits free hosting tiers.

## Run it locally

```bash
git clone https://github.com/sac23nht/Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS-.git
cd Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS-
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project folder (it is git-ignored):

```ini
PINECONE_API_KEY = "your-pinecone-key"
# Free token from https://huggingface.co/settings/tokens
# (fine-grained, tick "Make calls to Inference Providers").
HF_TOKEN = "hf_your-token"
```

Build the Pinecone index **once** (the app cannot answer anything until it exists). This is the
only step that needs the heavy libraries (torch):

```bash
pip install -r requirements-indexing.txt
python src/store_index.py
```

Start the app:

```bash
streamlit run app.py
```

> The index stores each passage's page number. If you built your index before that was added,
> re-run `python src/store_index.py` on a fresh index to get "PDF page N" in the Sources panel.

## Configuration

Set these as environment variables, in `.env`, or in Streamlit secrets:

| Variable | Required | Notes |
|---|---|---|
| `PINECONE_API_KEY` | yes | The index must already be populated |
| `HF_TOKEN` | yes | Hugging Face token with "Make calls to Inference Providers" |
| `LLM_MODEL` | no | Default `meta-llama/Llama-3.1-8B-Instruct` |
| `LLM_BASE_URL` | no | Default `https://router.huggingface.co/v1`. Any OpenAI-compatible API works (Groq, OpenRouter, OpenAI, ...) |
| `LLM_API_KEY` | no | Key for `LLM_BASE_URL`; defaults to `HF_TOKEN` |
| `PINECONE_INDEX_NAME` | no | Default `medical-chatbot` |
| `LINKEDIN_URL`, `PORTFOLIO_URL` | no | Adds buttons to the Developer section (must start with `https://`) |
| `DEVELOPER_NAME`, `GITHUB_URL`, `REPO_URL` | no | Override the name and links shown in the app |

## Deploy

### Streamlit Community Cloud (free)
1. New app > pick this repo, branch `main`, main file path `app.py`.
2. **Advanced settings > Python version: 3.12** (the default may be too new for some packages).
3. Advanced settings > **Secrets**:
   ```toml
   PINECONE_API_KEY = "your-pinecone-key"
   HF_TOKEN = "hf_your-token"
   ```

### Docker (Render, Railway, Fly.io, any container host)
```bash
docker build -t medibot .
docker run -p 8501:8501 -e PINECONE_API_KEY=... -e HF_TOKEN=... medibot
```
On **Render**: New Web Service > Language *Docker* > Instance type *Free* > add the two
variables under Environment > set the health check path to `/_stcore/health`. Free instances
sleep when idle, so the first request after a pause can take about a minute.

The free Hugging Face token has a limited monthly allowance for hosted inference. For heavier use,
point `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` at another provider.

## Project structure

```
├── app.py                      # Streamlit app (UI + RAG chain)
├── src/
│   ├── helper.py               # PDF loading, chunking, embeddings
│   ├── prompt.py               # System prompt
│   └── store_index.py          # One-off: build the Pinecone index
├── data/Medical_book.pdf       # Source book used to build the index
├── research/trials.ipynb       # Early experiments
├── requirements.txt            # App dependencies (lightweight)
├── requirements-indexing.txt   # Extra dependencies for building the index
├── Dockerfile
└── .streamlit/config.toml      # Streamlit theme and server settings
```

## Author

**Saijaya Rami Reddy Chilekampalli**: [GitHub](https://github.com/sac23nht)
