from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
import os


# Extract Data From the PDF File
def load_pdf_file(data: str) -> List[Document]:
    """
    Given a directory path, loads all PDF documents and returns the extracted documents.
    """
    if not os.path.isdir(data):
        raise ValueError(f"The directory {data} does not exist or is not a valid directory.")
    
    loader = DirectoryLoader(data, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, returns a new list containing only 'source' in metadata
    and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        if src:  # Make sure 'source' exists in the metadata
            minimal_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata={"source": src}
                )
            )
    return minimal_docs


# Split the Data into Text Chunks
def text_split(extracted_data: List[Document]) -> List[Document]:
    """
    Given a list of extracted document data, splits it into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.split_documents(extracted_data)


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensions


# Local embeddings, used by src/store_index.py to build the index.
# Needs torch + sentence-transformers (see requirements-indexing.txt).
def download_hugging_face_embeddings():
    """
    Downloads and returns the HuggingFace embeddings model, running locally.
    """
    from langchain_huggingface import HuggingFaceEmbeddings  # heavy: imported lazily

    try:
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception as e:
        raise RuntimeError(f"Failed to download HuggingFace embeddings: {e}")


# Hosted embeddings, used by the web app at query time. Same model as above, so
# the vectors match the ones already stored in Pinecone, but nothing is loaded
# into this process's memory.
def get_api_embeddings(token: str):
    from langchain_huggingface import HuggingFaceEndpointEmbeddings

    return HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL,
        huggingfacehub_api_token=token,
    )
