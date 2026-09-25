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

st.set_page_config(page_title="Medical Chatbot · RAG Q&A", page_icon="🩺", layout="wide")

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
PANEL_HEIGHT = 540  # pixel height of the chat panel (the info panel is a bit taller)
INDEXED_CHUNKS = "5,859"  # chunks stored in Pinecone by src/store_index.py
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
# Right panel: project details, source book and tool workflow
# ----------------------------
def flow_chart(nodes: list) -> None:
    """A simple top-to-bottom flow diagram; each node is 'line 1\nline 2'."""
    chain = " -> ".join(f'"{n}"' for n in nodes)
    st.graphviz_chart(
        'digraph { rankdir=TB; ranksep=0.22; bgcolor="transparent"; '
        'node [shape=box, style="rounded,filled", fillcolor="#0f766e", fontcolor="white", '
        'color="#0f766e", fontname="Helvetica", fontsize=11, margin="0.14,0.07"]; '
        f'edge [color="#64748b"]; {chain}; }}',
        width="stretch",
    )


def render_project_panel() -> None:
    model = pretty_model(LLM_MODEL)

    st.subheader("📌 About this project")
    st.markdown(
        "A **retrieval-augmented generation (RAG)** chatbot. It first looks up the most relevant "
        "passages in a medical encyclopedia, then has a language model write a short answer using "
        "only those passages. Every answer lists the passages it used, so you can check it."
    )
    st.markdown(
        f"**Developed by:** {DEVELOPER_NAME}  \n"
        f"**Language model:** {model}  \n"
        "**Embeddings:** all-MiniLM-L6-v2 (384 dimensions)"
    )

    st.subheader("📖 Source book")
    st.markdown(
        "**The Gale Encyclopedia of Medicine, 2nd edition** (2002)  \n"
        "Editor: Jacqueline L. Longe · Publisher: Gale Group  \n"
        "File used: `data/Medical_book.pdf`"
    )
    st.markdown(
        "This is the book that was read, embedded and stored in Pinecone:\n"
        f"- split into **{INDEXED_CHUNKS} text chunks** (~500 characters each)\n"
        "- embedded with **all-MiniLM-L6-v2**\n"
        f"- stored in the Pinecone index **{INDEX_NAME}** (cosine similarity)"
    )

    st.subheader("🛠️ Workflow")
    tab_setup, tab_query = st.tabs(["① One-time setup", "② Every question"])
    with tab_setup:
        flow_chart([
            "Medical_book.pdf\n(Gale Encyclopedia of Medicine)",
            "Load the pages\n(PyPDF via LangChain)",
            "Split into chunks\n(500 characters, 20 overlap)",
            "Embed each chunk\n(MiniLM-L6-v2, 384 numbers)",
            "Store the vectors\n(Pinecone index)",
        ])
    with tab_query:
        flow_chart([
            "Your question",
            "Embed the question\n(MiniLM-L6-v2, Hugging Face API)",
            "Search Pinecone\n(3 closest chunks)",
            f"Write the answer\n({model}, only from those chunks)",
            "Answer + Sources\n(shown in the chat)",
        ])

    st.subheader("🧰 Tools")
    st.markdown(
        "| Step | Tool |\n|---|---|\n"
        "| Read the PDF | PyPDF (LangChain loader) |\n"
        "| Split the text | LangChain text splitter |\n"
        "| Embeddings | Sentence-Transformers MiniLM-L6-v2 |\n"
        "| Vector database | Pinecone |\n"
        f"| Language model | {model} (Hugging Face Inference API) |\n"
        "| Web app | Python, Streamlit |\n"
        "| Packaging | Docker |"
    )

    st.subheader("👩‍💻 Developer")
    st.markdown(f"**{DEVELOPER_NAME}** designed, built and deployed this project end to end.")
    links = [(label, url) for label, url in (
        ("GitHub profile", GITHUB_URL), ("Source code", REPO_URL),
        ("LinkedIn", LINKEDIN_URL), ("Portfolio", PORTFOLIO_URL)) if url]
    for row in range(0, len(links), 2):
        cols = st.columns(2)
        for col, (label, url) in zip(cols, links[row:row + 2]):
            col.link_button(label, url, width="stretch")


# ----------------------------
# Page
# ----------------------------
st.title("🩺 Medical Chatbot")
st.markdown(
    "Ask a medical question and get a short answer **grounded in a real reference book**, "
    "with the passages it used shown as sources."
)
st.caption(f"Developed by **{DEVELOPER_NAME}** · LangChain · Pinecone · {pretty_model(LLM_MODEL)} · Streamlit")

if "messages" not in st.session_state:
    st.session_state.messages = []

chat_col, info_col = st.columns([3, 2], gap="large")

with info_col:
    with st.container(height=PANEL_HEIGHT + 70, border=True):
        render_project_panel()

with chat_col:
    missing = [n for n, v in (("PINECONE_API_KEY", PINECONE_API_KEY), ("HF_TOKEN", HF_TOKEN)) if not v]
    if missing:
        st.error(f"Missing configuration: {', '.join(missing)}. Set it as an environment variable, "
                 "in a local `.env` file, or in Streamlit secrets.")
        st.stop()

    chat_box = st.container(height=PANEL_HEIGHT, border=True)
    typed = st.chat_input("Ask a medical question…", max_chars=MAX_QUESTION_CHARS)

question = (st.session_state.pop("pending", None) or typed or "").strip()

with chat_box:
    if not st.session_state.messages and not question:
        with st.chat_message("assistant"):
            st.markdown("Hi! Ask me a medical question in plain English, for example about a "
                        "symptom, condition or treatment. I answer from the *Gale Encyclopedia of "
                        "Medicine* and show my sources.")
        st.caption("Try an example:")
        example_cols = st.columns(2)
        for i, example in enumerate(EXAMPLES):
            example_cols[i % 2].button(
                example, key=f"ex-{i}", width="stretch",
                on_click=lambda q=example: st.session_state.update(pending=q),
            )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.text(message["content"])
            else:
                st.markdown(message["content"])
                render_sources(message.get("sources"))

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.text(question)
        with st.chat_message("assistant"):
            try:
                chain = load_chain(PINECONE_API_KEY, HF_TOKEN, INDEX_NAME,
                                   LLM_BASE_URL, LLM_MODEL, LLM_API_KEY)
                with st.spinner("Searching the encyclopedia…"):
                    response = chain.invoke({"input": question})
                answer = (response.get("answer") or "").strip() or "Sorry, I couldn't generate an answer."
                sources = format_sources(response.get("context", []))
            except Exception as exc:  # network / quota / index problems
                print("Error:", exc)
                answer = "Sorry, something went wrong while generating the answer. Please try again."
                sources = []
            st.markdown(answer)
            render_sources(sources)
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

st.caption("Educational demo, not medical advice. Answers are AI-generated from the "
           "Gale Encyclopedia of Medicine (2nd ed., 2002) and may be incomplete or out of date.")
