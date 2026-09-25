"""Medical Chatbot: a RAG question-answering app built with Streamlit.

Flow for every question: embed it (hosted MiniLM) -> retrieve the 3 closest
passages from Pinecone -> ask a hosted LLM to answer using only those passages
-> show the answer together with the passages it was based on.
"""
import os
import re

import streamlit as st
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from src.helper import get_api_embeddings
from src.prompt import system_prompt

load_dotenv()

st.set_page_config(page_title="Medical Chatbot · RAG Q&A", page_icon="🩺", layout="centered")

# ----------------------------
# Settings (environment variables, a local .env file, or Streamlit secrets)
# ----------------------------
def get_setting(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except Exception:  # no secrets file configured
        return default


def safe_url(value: str) -> str:
    """Only http(s) links are ever rendered as buttons."""
    return value if re.match(r"^https?://", value or "") else ""


PINECONE_API_KEY = get_setting("PINECONE_API_KEY")
HF_TOKEN = get_setting("HF_TOKEN")
INDEX_NAME = get_setting("PINECONE_INDEX_NAME", "medical-chatbot")
# Any OpenAI-compatible chat API (Hugging Face router by default).
LLM_BASE_URL = get_setting("LLM_BASE_URL", "https://router.huggingface.co/v1")
LLM_MODEL = get_setting("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
LLM_API_KEY = get_setting("LLM_API_KEY") or HF_TOKEN

DEVELOPER_NAME = get_setting("DEVELOPER_NAME", "Saijaya Rami Reddy Chilekampalli")
GITHUB_URL = safe_url(get_setting("GITHUB_URL", "https://github.com/sac23nht"))
REPO_URL = safe_url(get_setting(
    "REPO_URL", "https://github.com/sac23nht/Medical-Chatbot-with-LLMs-LangChain-Pinecone-Flask-AWS-"
))
LINKEDIN_URL = safe_url(get_setting("LINKEDIN_URL"))
PORTFOLIO_URL = safe_url(get_setting("PORTFOLIO_URL"))

# Friendly names for the file names stored in the Pinecone metadata.
SOURCE_TITLES = {"Medical_book.pdf": "The Gale Encyclopedia of Medicine, 2nd ed. (2002)"}
MAX_QUESTION_CHARS = 500
EXAMPLES = [
    "What are the symptoms of diabetes?",
    "What is acne and how is it treated?",
    "What causes high blood pressure?",
    "What is anemia?",
]


def pretty_model(name: str) -> str:
    """'meta-llama/Llama-3.1-8B-Instruct' -> 'Llama 3.1 8B'."""
    base = re.sub(r"[-_ ]?instruct$", "", name.split("/")[-1], flags=re.IGNORECASE)
    return re.sub(r"[-_]", " ", base)


# ----------------------------
# RAG chain (built once per server process)
# ----------------------------
@st.cache_resource(show_spinner="Connecting to the knowledge base…")
def load_chain(pinecone_key: str, hf_token: str, index_name: str,
               base_url: str, model: str, llm_key: str):
    os.environ["PINECONE_API_KEY"] = pinecone_key  # read by langchain-pinecone
    docsearch = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=get_api_embeddings(hf_token),
    )
    retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    llm = ChatOpenAI(
        model=model, base_url=base_url, api_key=llm_key,
        temperature=0.2, max_tokens=300, timeout=60, max_retries=2,
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    return create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))


def shorten(text: str, limit: int = 260) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def format_sources(docs) -> list:
    """Turn retrieved LangChain documents into a small list for display."""
    sources, seen = [], set()
    for doc in docs:
        text = (doc.page_content or "").replace("\N{REPLACEMENT CHARACTER}", "")
        text = re.sub(r"\s+", " ", text).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        filename = os.path.basename(str(doc.metadata.get("source", "")).replace("\\", "/"))
        item = {"title": SOURCE_TITLES.get(filename, filename or "Unknown source"),
                "excerpt": shorten(text)}
        page = doc.metadata.get("page")  # only present in indexes built with page numbers
        if isinstance(page, (int, float)):
            item["page"] = int(page) + 1  # PyPDFLoader pages are 0-based
        sources.append(item)
    return sources


def escape_md(text: str) -> str:
    """Escape markdown characters so a short title shows literally inside st.markdown."""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|<>~$&])", r"\\\1", text)


def render_sources(sources: list) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources ({len(sources)})"):
        st.caption("Passages retrieved from the reference book for this answer.")
        for i, src in enumerate(sources, 1):
            page = f", PDF page {src['page']}" if "page" in src else ""
            st.markdown(f"**{i}. {escape_md(src['title'])}{page}**")
            st.text(src["excerpt"])  # plain text: never parsed as markdown, so no links or HTML


# ----------------------------
# Header
# ----------------------------
st.title("🩺 Medical Chatbot")
st.markdown(
    "Ask a medical question and get a short answer **grounded in a real reference book**, "
    "the *Gale Encyclopedia of Medicine*, with the passages it used shown as sources."
)
st.caption(f"Developed by **{DEVELOPER_NAME}** · LangChain · Pinecone · {pretty_model(LLM_MODEL)} · Streamlit")

c1, c2, c3 = st.columns(3)
c1.metric("Passages indexed", "~5,900")
c2.metric("Retrieved per question", "3")
c3.metric("Language model", pretty_model(LLM_MODEL))

if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# About (open for first-time visitors, folds away once the chat starts)
# ----------------------------
with st.expander("ℹ️ What is this, how do I use it, and who built it?",
                 expanded=not st.session_state.messages):
    t_about, t_use, t_how, t_tech, t_dev = st.tabs(
        ["What is it", "How to use", "How it works", "Built with", "Developer"]
    )
    with t_about:
        st.markdown(
            "A **retrieval-augmented generation (RAG)** chatbot. Rather than answering from "
            "memory, it first *looks up* the most relevant passages in a medical encyclopedia, "
            "then has a language model write a short answer using only those passages. "
            "Every answer lists the passages it was based on, so you can check it."
        )
        st.markdown(
            "- **Grounded:** the model is told to say it doesn't know if the passages don't cover the question.\n"
            "- **Transparent:** open **Sources** under any answer to see the retrieved text.\n"
            "- **Lightweight:** embeddings and the language model are hosted APIs, "
            "so the app needs no GPU and only a few hundred MB of RAM."
        )
        st.info("Educational demo, **not medical advice**. Answers come from an AI model reading a "
                "2002 reference book and may be incomplete or out of date.")
    with t_use:
        st.markdown(
            "1. **Ask**: type a question in plain English below, or tap an example.\n"
            "2. **Read**: you get a short answer written from the top three matching passages.\n"
            "3. **Check**: expand **📚 Sources** to see the exact passages behind the answer.\n"
            "4. **Refine**: ask about one condition or symptom at a time, and rephrase if the answer is thin."
        )
        st.caption("The first reply after a quiet period can take up to a minute on a free host.")
    with t_how:
        st.graphviz_chart(
            'digraph { rankdir=TB; ranksep=0.25; bgcolor="transparent"; '
            'node [shape=box, style="rounded,filled", fillcolor="#0f766e", fontcolor="white", '
            'color="#0f766e", fontname="Helvetica", fontsize=11]; edge [color="#64748b"]; '
            '"Your question" -> "Embed\\n(MiniLM-L6-v2)" -> "Retrieve top 3\\n(Pinecone)" '
            f'-> "Generate\\n({pretty_model(LLM_MODEL)})" -> "Answer + sources"; }}',
            width="stretch",
        )
        st.markdown(
            "1. Your question is turned into a 384-number vector by the MiniLM-L6-v2 embedding model.\n"
            "2. Pinecone finds the 3 most similar passages among ~5,900 chunks of the book.\n"
            "3. The language model writes a short answer using only those passages.\n"
            "4. The answer and its sources are shown here."
        )
        st.caption("Setup, done once: the book's PDF is split into ~500-character chunks, embedded, "
                   "and stored in a Pinecone index.")
    with t_tech:
        st.markdown(
            "**AI & retrieval:** RAG · LangChain · Pinecone · Sentence-Transformers (MiniLM-L6-v2) · "
            f"{pretty_model(LLM_MODEL)} · Hugging Face Inference API  \n"
            "**App:** Python · Streamlit  \n"
            "**Data pipeline:** PyPDF · text chunking  \n"
            "**Deployment:** Docker · Git & GitHub"
        )
    with t_dev:
        st.subheader(DEVELOPER_NAME)
        st.markdown(
            "Designed, built and deployed this project end to end: document ingestion and chunking, "
            "the vector index, the RAG pipeline, this interface, containerisation and hosting."
        )
        cols = st.columns(4)
        buttons = [("GitHub profile", GITHUB_URL), ("Source code", REPO_URL),
                   ("LinkedIn", LINKEDIN_URL), ("Portfolio", PORTFOLIO_URL)]
        for col, (label, url) in zip(cols, [b for b in buttons if b[1]] + [("", "")] * 4):
            if url:
                col.link_button(label, url)

# ----------------------------
# Chat
# ----------------------------
missing = [n for n, v in (("PINECONE_API_KEY", PINECONE_API_KEY), ("HF_TOKEN", HF_TOKEN)) if not v]
if missing:
    st.error(f"Missing configuration: {', '.join(missing)}. Set it as an environment variable, "
             "in a local `.env` file, or in Streamlit secrets.")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.text(message["content"])
        else:
            st.markdown(message["content"])
            render_sources(message.get("sources"))

if not st.session_state.messages:
    st.markdown("**Try an example:**")
    for example in EXAMPLES:
        st.button(example, key=f"ex-{example}", on_click=lambda q=example: st.session_state.update(pending=q))

typed = st.chat_input("Ask a medical question…", max_chars=MAX_QUESTION_CHARS)
question = (st.session_state.pop("pending", None) or typed or "").strip()

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.text(question)
    with st.chat_message("assistant"):
        try:
            chain = load_chain(PINECONE_API_KEY, HF_TOKEN, INDEX_NAME, LLM_BASE_URL, LLM_MODEL, LLM_API_KEY)
            with st.spinner("Searching the encyclopedia…"):
                response = chain.invoke({"input": question})
            answer = (response.get("answer") or "").strip() or "Sorry, I couldn't generate an answer."
            sources = format_sources(response.get("context", []))
        except Exception as exc:  # network / quota / index problems
            print("Error:", exc)
            answer, sources = "Sorry, something went wrong while generating the answer. Please try again.", []
        st.markdown(answer)
        render_sources(sources)
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    if len(st.session_state.messages) == 2:
        st.rerun()  # fold the About section now that the conversation has started

st.caption("Educational demo, not medical advice. Answers are AI-generated from the "
           "Gale Encyclopedia of Medicine (2nd ed., 2002) and may be incomplete or out of date.")
