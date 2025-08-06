# utils.py
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import os
from models.embeddings import get_embedding_model # Import the embedding model
from googleapiclient.discovery import build
from config.config import GOOGLE_SEARCH_API_KEY, GOOGLE_CSE_ID

def get_pdf_text(pdf_docs):
    """
    Extracts text from a list of PDF documents.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """
    Splits a given text into smaller, manageable chunks.
    """
    # Adjusted chunk_size and chunk_overlap for better performance and context
    # 50000 is very large; typically chunks are much smaller (e.g., 1000-2000)
    # For academic papers, you might need larger chunks, but be mindful of LLM context window limits.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=2000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """
    Creates a FAISS vector store from text chunks and saves it locally.
    """
    embeddings = get_embedding_model() # Use the imported embedding model
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def load_vector_store():
    """
    Loads the FAISS vector store from local storage.
    """
    embeddings = get_embedding_model() # Use the imported embedding model for loading
    # IMPORTANT: Set allow_dangerous_deserialization=True as per previous error fix
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    return new_db

def perform_web_search(query):
    """
    Performs a real-time web search using Google Custom Search API.
    Returns a list of snippets.
    """
    if not GOOGLE_SEARCH_API_KEY or not GOOGLE_CSE_ID:
        print("Google Search API Key or CSE ID not configured.")
        return []

    try:
        service = build("customsearch", "v1", developerKey=GOOGLE_SEARCH_API_KEY)
        res = service.cse().list(q=query, cx=GOOGLE_CSE_ID, num=3).execute() # num=3 for top 3 results

        snippets = []
        if 'items' in res:
            for item in res['items']:
                snippets.append(item.get('snippet'))
        return snippets
    except Exception as e:
        print(f"Error performing web search: {e}")
        return []