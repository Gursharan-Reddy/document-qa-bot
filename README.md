# Multi-Document Interactive Q&A Bot Engine (RAG Pipeline)

An advanced, production-ready Retrieval-Augmented Generation (RAG) pipeline designed to ingest local textual artifacts, vectorize them into a persistent database, and provide a secure, interactive command-line interface for context-grounded conversational question answering. 

This engine is powered by the modern Google GenAI SDK (`google-genai`), utilizing `gemini-embedding-2` for multi-axis vector space coordinate alignment and `gemini-2.5-flash` for deterministic, citation-accurate text synthesis.

---

## 🏗️ System Architecture & Mechanics

The codebase is engineered strictly around modular separation of concerns:

```text
├── data/                  # Raw document ingestion boundary (.pdf, .docx, .txt)
├── db/                    # Local persistent ChromaDB vector storage engine
├── src/
│   ├── __init__.py        # Module namespace initializer
│   ├── config.py          # Central environment and parameters manifest
│   ├── ingest.py          # Extraction, dynamic chunking, and backoff indexing pipeline
│   ├── query.py           # Coordinate mapping retrieval and grounded inference engine
│   └── main.py            # REPL Command Line Interface orchestration layer
├── .env                   # Protected environment variables storage
├── .gitignore             # Deployment tracking exclusions rules
└── requirements.txt       # Unified Python package dependencies manifest

Core Implementation Highlights
Mathematical Vector Alignment: Resolves implicit dimensional coordinate fragmentation bugs by manually calculating text query shapes before execution, forcing strict spatial compatibility inside ChromaDB collections.

Recursive Sliding-Window Chunking: Chunks text blocks using explicit parameter bounds (1000 characters with a 200-character overlap), preventing structural sentence splitting and safeguarding localized semantics.

Resilient Production Backoff: Features a multi-tiered retry and pause abstraction strategy within indexing loops and response generations to gracefully navigate high-demand traffic phases (such as 503 Service Unavailable or 429 Rate Limits) on the Gemini API Free Tier.

Strict Hallucination Grounding: Configured with explicit strict-system prompts ensuring that the LLM denies answers using a standard fallback response sequence if facts do not exist inside retrieved document frames.

🛠️ Installation & Setup
1. Environment Activation
Ensure you are running inside a pristine Python 3.13 virtual environment wrapper. Activate it using PowerShell:
# Create environment if not already initialized
python -m venv venv

# Activate active script boundaries
venv\Scripts\activate

2. Dependency Manifest Ingestion
Install all required packages explicitly handled via pip matching your target environment:

pip install -r requirements.txt

3. Environment Variable Configuration
Create a .env file in the root directory and append your secure API authorization credentials:

Code snippet
GEMINI_API_KEY="GEMINI_API_KEY"
(Note: Ensure your .gitignore contains rules blocking .env and db/ from accidentally getting exposed on remote public git branches!)

🚀 Execution & Runtime Cycle
Phase 1: Knowledge-Base Ingestion & Cache Compilation
Place your source documents (e.g., project guidelines, booklets, manuals) into the data/ folder. Execute the text parser and local indexer:

python src/ingest.py
The script will systematically extract the strings, divide them into structural pieces, call the embedding models, and cache a persistent matrix on your local machine under db/ showing a progress completion rate from 0% to 100%.

Phase 2: Interactive Query Execution Runtime
Boot up the command-line REPL interaction interface:

python src/main.py
You can now enter natural language queries. The engine will retrieve relevant text fragments, pass them securely to Gemini, and synthesize answers coupled with granular source tracking. To close out the active loop session, type exit or quit.

📋 Recommended Core Proof-of-Concept Test Suite
To thoroughly demonstrate the system's cross-document accuracy and safety guardrails, execute the following questions sequence:

Process & Domain Execution Frameworks:

Query: What are the core responsibilities of a Scrum Master?

Target Source Resolution: Scrum Alliance Framework Insights_removed.pdf

Technical Data & Architecture Reference Compilation:

Query: What is the purpose of the Property Management System reference design?

Target Source Resolution: NASA Climate Change Educational Guide_removed.pdf

Operational Rules Analysis:

Query: What sections are required in the Executive Summary of the business plan?

Target Source Resolution: sample_business_plan_removed.pdf

Safety & Grounding Guardrail Test (Unanswerable Verification):

Query: What is the company's policy on deploying quantum computing software algorithms?

Expected Safe Outcome: I am sorry, but the provided documents do not contain the answer to your question. (Proves zero hallucination parameters).

⚖️ Git Development Version-Control Footprint
This codebase contains a structured commit history tracking the evolution of optimization updates, ensuring a professional, auditable repository timeline:

feat: complete robust local RAG pipeline using google-genai and chromadb — Baseline initialization of files architecture structure.

refactor: optimize query pipeline with manual embedding execution and server retry logic — Implementation of custom structural models configuration mapping and transient error management logic.