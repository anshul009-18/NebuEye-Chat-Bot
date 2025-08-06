# models/llm.py
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

load_dotenv() # Load environment variables

def get_chat_model(model_name="gemini-1.5-flash", temperature=0.3):
    """
    Initializes and returns the Google Generative AI Chat Model.
    """
    # Ensure GOOGLE_API_KEY is loaded from .env and passed correctly
    # The API key is usually picked up automatically by langchain_google_genai
    # if set as GOOGLE_API_KEY environment variable.
    return ChatGoogleGenerativeAI(model=model_name, temperature=temperature)