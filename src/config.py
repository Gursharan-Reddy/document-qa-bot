import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Fix: Update model to the standard model identifier for the new GenAI SDK
EMBEDDING_MODEL = "gemini-embedding-2"
LLM_MODEL = "gemini-2.5-flash"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_DIR = os.path.join(BASE_DIR, "db")

COLLECTION_NAME = "document_knowledge_base"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

if not GEMINI_API_KEY:
    raise ValueError("CRITICAL ERROR: GEMINI_API_KEY is missing from your .env file.")