import os
import sys
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import chromadb
from pypdf import PdfReader
from docx import Document
from tqdm import tqdm
from google import genai
from google.genai import types
from chromadb.api.types import EmbeddingFunction, Documents, Embeddings
from src import config

class GeminiModernEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str, model_name: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def __call__(self, input: Documents) -> Embeddings:
        formatted_contents = [
            types.Content(parts=[types.Part.from_text(text=text_chunk)])
            for text_chunk in input
        ]
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=formatted_contents
        )
        return [e.values for e in response.embeddings]

def extract_text_from_pdf(file_path: str) -> list[dict]:
    extracted_data = []
    file_name = os.path.basename(file_path)
    try:
        reader = PdfReader(file_path)
        for index, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                clean_text = " ".join(text.split())
                extracted_data.append({
                    "text": clean_text,
                    "metadata": {
                        "source": file_name,
                        "page": str(index + 1)
                    }
                })
    except Exception as e:
        print(f"Error reading PDF {file_name}: {e}")
    return extracted_data

def extract_text_from_docx(file_path: str) -> list[dict]:
    extracted_data = []
    file_name = os.path.basename(file_path)
    try:
        doc = Document(file_path)
        full_text = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
        combined_text = "\n".join(full_text)
        if combined_text:
            clean_text = " ".join(combined_text.split())
            extracted_data.append({
                "text": clean_text,
                "metadata": {
                    "source": file_name,
                    "page": "N/A (Word Doc)"
                }
            })
    except Exception as e:
        print(f"Error reading DOCX {file_name}: {e}")
    return extracted_data

def chunk_documents(pages: list[dict]) -> list[dict]:
    chunks = []
    for page in pages:
        text = page["text"]
        metadata = page["metadata"]
        start = 0
        text_length = len(text)
        while start < text_length:
            end = min(start + config.CHUNK_SIZE, text_length)
            chunk_text = text[start:end]
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": metadata["source"],
                    "page": metadata["page"],
                    "chunk_range": f"{start}-{end}"
                }
            })
            start += (config.CHUNK_SIZE - config.CHUNK_OVERLAP)
    return chunks

def build_vector_pipeline():
    print("Initializing Ingestion Pipeline Engine...")
    
    if not os.path.exists(config.DATA_DIR) or not os.listdir(config.DATA_DIR):
        print(f"Aborted: Please drop documents inside the '{config.DATA_DIR}/' folder first!")
        return

    all_pages = []
    for file in os.listdir(config.DATA_DIR):
        file_path = os.path.join(config.DATA_DIR, file)
        if file.lower().endswith('.pdf'):
            all_pages.extend(extract_text_from_pdf(file_path))
        elif file.lower().endswith('.docx'):
            all_pages.extend(extract_text_from_docx(file_path))
        elif file.lower().endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                all_pages.append({
                    "text": " ".join(f.read().split()),
                    "metadata": {"source": file, "page": "1"}
                })

    print(f"Extracted {len(all_pages)} structural items. Splitting into overlapping chunks...")
    chunks = chunk_documents(all_pages)
    print(f"Processed {len(chunks)} contextual snippets.")

    chroma_client = chromadb.PersistentClient(path=config.DB_DIR)
    embedding_function = GeminiModernEmbeddingFunction(
        api_key=config.GEMINI_API_KEY,
        model_name=config.EMBEDDING_MODEL
    )
    
    collection = chroma_client.get_or_create_collection(
        name=config.COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"}
    )

    ids = [f"doc_chunk_{i}" for i in range(len(chunks))]
    documents = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    print("Generating embeddings and storing locally in ChromaDB...")
    
    BATCH_SIZE = 25
    for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
        batch_end = min(i + BATCH_SIZE, len(chunks))
        collection.add(
            ids=ids[i:batch_end],
            documents=documents[i:batch_end],
            metadatas=metadatas[i:batch_end]
        )
        time.sleep(6)

    print(f"Successfully indexed and cached vector knowledge base at '{config.DB_DIR}'!")

if __name__ == "__main__":
    build_vector_pipeline()