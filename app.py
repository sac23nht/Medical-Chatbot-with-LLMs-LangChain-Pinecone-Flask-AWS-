from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain.llms.base import LLM
from typing import Optional, List
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
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
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# ----------------------------
# Pinecone VectorStore
# ----------------------------
embeddings = download_hugging_face_embeddings()
index_name = "medical-chatbot"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# ----------------------------
# Hugging Face LLM for RAG
# ----------------------------
class HFAPIModel(LLM):
    model_name: str = "google/flan-t5-base"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Load tokenizer and model once at startup
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.pipe = pipeline(
            "text2text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=256,
            device=-1  # CPU; set to 0 for GPU
        )

    @property
    def _llm_type(self) -> str:
        return "hf-api"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        result = self.pipe(prompt)
        return result[0]["generated_text"]

# Initialize the LLM
chatModel = HFAPIModel()

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
    app.run(host="0.0.0.0", port=8080, debug=True)
