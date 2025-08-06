# config/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# Google API Key for Generative AI (already in .env)
# GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Google Custom Search API Key and CX (Engine ID)
# You'll need to enable Custom Search API in Google Cloud Console
# and create a Custom Search Engine.
GOOGLE_SEARCH_API_KEY = os.getenv("GOOGLE_SEARCH_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

# Example for other API keys if you use different services
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# GROQ_API_KEY = os.getenv("GROQ_API_KEY")