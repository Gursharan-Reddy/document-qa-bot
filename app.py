import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.query import execute_rag_query

app = FastAPI(title="Knowledge Matrix RAG API Engine")

class QueryRequest(BaseModel):
    question: str
    k: int = 8

@app.post("/api/query")
def run_query(request: QueryRequest):
    try:
        result = execute_rag_query(request.question, k=request.k)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "healthy", "engine": "RAG Knowledge Matrix"}