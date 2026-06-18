import sys
import os
from query import execute_rag_query

def launch_cli_interface():
    print("=" * 60)
    print("KNOWLEDGE MATRIX: INTERACTIVE DOCUMENT Q&A BOT BOT")
    print("=" * 60)
    print("Instructions: Enter your document query strings below.")
    print("Type 'exit' or 'quit' to terminate the runtime cycle.\n")

    while True:
        try:
            user_query = input("\nAsk a question: ").strip()
            if not user_query:
                continue
            if user_query.lower() in ['exit', 'quit']:
                print("\nShutting down Q&A runtime. Happy hacking! 🚀")
                break

            print("Fetching document context blocks and parsing with Gemini...")
            result = execute_rag_query(user_query, k=8)

            print("\n" + "ANSWER:" + "\n" + "-" * 40)
            print(result["answer"])
            print("-" * 40)

            if result["citations"]:
                print("\nRETRIEVED SOURCE CITATIONS:")
                for citation in result["citations"]:
                    print(f" • {citation}")
            print("=" * 60)

        except KeyboardInterrupt:
            print("\nSession aborted cleanly.")
            break
        except Exception as e:
            print(f"\nPipeline Runtime Error encountered: {e}")

if __name__ == "__main__":
    launch_cli_interface()