from google import genai
from google.genai import types
import chromadb
import config
from src import config

class GeminiModernEmbeddingFunction:
    def __init__(self, api_key: str, model_name: str):
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name

    def __call__(self, input: list[str]) -> list[list[float]]:
        formatted_contents = [
            types.Content(parts=[types.Part.from_text(text=text_chunk)])
            for text_chunk in input
        ]
        response = self.client.models.embed_content(
            model=self.model_name,
            contents=formatted_contents
        )
        return [e.values for e in response.embeddings]

def execute_rag_query(user_query: str, k: int = 4) -> dict:
    chroma_client = chromadb.PersistentClient(path=config.DB_DIR)
    
    embedder = GeminiModernEmbeddingFunction(
        api_key=config.GEMINI_API_KEY,
        model_name=config.EMBEDDING_MODEL
    )

    try:
        collection = chroma_client.get_collection(name=config.COLLECTION_NAME)
    except Exception:
        return {
            "answer": "Error: Vector database cache directory was not found. Please run ingestion first.",
            "citations": [], "raw_context": []
        }

    query_embeddings = embedder([user_query])

    results = collection.query(
        query_embeddings=query_embeddings,
        n_results=k
    )

    if not results or not results['documents'] or not results['documents'][0]:
        return {
            "answer": "I am sorry, but no context modules could be safely retrieved.",
            "citations": [], "raw_context": []
        }

    context_blocks = []
    citations = []
    
    for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
        source_name = meta['source']
        page_num = meta['page']
        citation_str = f"Source: {source_name}, Page: {page_num}"
        context_blocks.append(f"[{citation_str}]\nContext Chunk: {doc}")
        citations.append(citation_str)

    context_payload = "\n\n---\n\n".join(context_blocks)

    system_prompt = (
        "You are a professional, accurate document Q&A assistant.\n"
        "Answer the user's question using ONLY the provided document context below.\n"
        "Cite your sources (filenames and page details) inline next to the facts you cite.\n"
        "If the answer cannot be found in the context, clearly state: "
        "'I am sorry, but the provided documents do not contain the answer to your question.'\n"
        "Do not invent facts or extrapolate using outside training knowledge."
    )

    full_prompt = (
        f"{system_prompt}\n\n"
        f"CONTEXT INFORMATION:\n{context_payload}\n\n"
        f"USER QUESTION: {user_query}\n\n"
        f"GROUNDED ANSWER:"
    )

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=config.LLM_MODEL,
        contents=full_prompt,
    )

    return {
        "answer": response.text,
        "citations": list(set(citations)),
        "raw_context": results['documents'][0]
    }