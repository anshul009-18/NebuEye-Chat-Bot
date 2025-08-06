# models/embeddings.py
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables from .env

def get_embedding_model():
    """
    Initializes and returns the Google Generative AI Embeddings model.
    """
    # Ensure GOOGLE_API_KEY is loaded from .env
    # The model name "models/embedding-001" is standard for Google's embedding model
    return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# You might also want to add a function here to handle model configuration
# if there were different embedding models to choose from.