from flask import Flask, render_template, request
from src.helper import get_api_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()


def require_env(name: str, hint: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. {hint} Add it to a .env file locally, or as an "
            "environment variable / secret on your hosting platform."
        )
    return value


PINECONE_API_KEY = require_env("PINECONE_API_KEY", "It is used to search the vector index.")
# One Hugging Face token is used for the hosted embedding model and, by default, the LLM.
HF_TOKEN = require_env("HF_TOKEN", "Create a free token at https://huggingface.co/settings/tokens.")

# LLM: any OpenAI-compatible chat endpoint (Hugging Face router by default;
# Groq, OpenRouter, OpenAI, ... work by changing these three variables).
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://router.huggingface.co/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
LLM_API_KEY = os.getenv("LLM_API_KEY") or HF_TOKEN

# ----------------------------
# Pinecone VectorStore
# ----------------------------
embeddings = get_api_embeddings(HF_TOKEN)
index_name = os.getenv("PINECONE_INDEX_NAME", "medical-chatbot")
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ----------------------------
# Hosted LLM for RAG
# ----------------------------
# The model runs on the provider's servers, so this app needs no GPU and only a
# few hundred MB of RAM (the previous local flan-t5-base needed ~1.3 GB).
chatModel = ChatOpenAI(
    model=LLM_MODEL,
    base_url=LLM_BASE_URL,
    api_key=LLM_API_KEY,
    temperature=0.2,
    max_tokens=300,
    timeout=60,
    max_retries=2,
)

# ----------------------------
# RAG setup
# ----------------------------
system_prompt = (
    "You are a medical assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ----------------------------
# Flask routes
# ----------------------------
@app.route("/health")
def health():
    return "ok", 200


@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "")
    print("User:", msg)
    try:
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "Error generating response.")
    except Exception as e:
        print("Error:", e)
        answer = "Error generating response."

    return answer

# ----------------------------
if __name__ == "__main__":
    # Local development only. In production the app is served by gunicorn
    # (see Dockerfile). debug is off by default: the debug reloader would load
    # the models twice and exposes an interactive debugger.
    port = int(os.getenv("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=os.getenv("FLASK_DEBUG") == "1")
