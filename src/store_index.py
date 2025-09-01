from dotenv import load_dotenv
import os
from helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec  # ✅ v3 SDK
from langchain_pinecone import PineconeVectorStore  # ✅ new adapter

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables.")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# Path setup
base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
pdf_folder = os.path.join(base_path, "data")

# Load PDF data
extracted_data = load_pdf_file(pdf_folder)
filter_data = filter_to_minimal_docs(extracted_data)
text_chunks = text_split(filter_data)

# Download embeddings
embeddings = download_hugging_face_embeddings()

# ✅ Initialize Pinecone (v3)
pc = Pinecone(api_key=PINECONE_API_KEY)

# Index name
index_name = "medical-chatbot"

# Create index if missing
if index_name not in [idx["name"] for idx in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,  # must match embedding size
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

# ✅ Use new LangChain integration
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name,
    namespace=None,
    pinecone_api_key=PINECONE_API_KEY,
)

print(f"Stored {len(text_chunks)} chunks in Pinecone.")
