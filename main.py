import os
import json
from src.index import index_directory
from src.llm_tool import call_llm 
from src.rag import RAGSystem
from dotenv import load_dotenv

prompt = """
You are a precise file locator. 
I will provide you with several text chunks, each with a unique Chunk ID and a Source File Path.
Your task is to identify which chunk contains the answer to the user's query.

FORMAT:
Return ONLY a valid JSON object with a single key "chunk_id".
Example: {"chunk_id": 42}
If no chunk is relevant, return {"chunk_id": null}.
Do not include any other text, markdown formatting, or explanations.
"""

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    output_dir = os.path.join(script_dir, 'output')
    embeddings_dir = os.path.join(output_dir, 'embeddings')

    os.makedirs(output_dir, exist_ok=True)

    if os.path.exists(data_dir):
        # 1. Load Data
        data_index = index_directory(data_dir)

        # 2. RAG System
        print("Initializing RAG system...")
        rag = RAGSystem(index_path=embeddings_dir)
        
        # Check if we need to ingest (simple check: if not loaded, ingest)
        if not rag.load_index():
            print("Indexing documents... (First run may be slow)")
            rag.ingest(data_index)
        else:
            print("Loaded existing RAG index.")

        load_dotenv()
        query = input("Enter your query: ")
        
        # 3. Retrieve relevant chunks (k=5 to give LLM options)
        results = rag.search(query, k=5, alpha=0.5)
        
        if not results:
            print("No relevant documents found.")
            return

        # We send structured JSON with ID, Path, and Content
        chunks_payload = []
        chunk_map = {} 
        
        print(f"Retrieved {len(results)} chunks.")
        
        for res in results:
            # We must handle cases where 'chunk_id' might be missing if rag.py hasn't been reloaded properly
            # But we edited rag.py, so it should be fine.
            c_id = res.get('chunk_id')
            if c_id is None:
                # Fallback if rag.py logic is stale in memory
                c_id = res.get('chunk', {}).get('id', -1)
                
            chunk_data = {
                "chunk_id": c_id,
                "source_file": res['path'],
                "content": res.get('content', '') # Content is directly in result now
            }
            chunks_payload.append(chunk_data)
            chunk_map[c_id] = res

        # 4. Call LLM with ONLY the relevant chunks
        context_str = json.dumps(chunks_payload, indent=2)
        
        print(f"Calling LLM to identify the correct chunk...")
        output_text = call_llm(
            api_key=os.getenv("LLM_KEY"),
            model="mistral/mistral-small-latest",
            messages=f"Query: \n {query} \n Prompt:{prompt}\n Chunks: \n{context_str}"
        )
        print("LLM response received:", output_text)
        
        # 5. Parse JSON output and find the file
        try:
            # Clean up potential markdown formatting from LLM
            clean_json = output_text.strip()
            if clean_json.startswith("```json"):
                clean_json = clean_json[7:]
                if clean_json.endswith("```"):
                     clean_json = clean_json[:-3]
            elif clean_json.startswith("```"):
                 clean_json = clean_json[3:]
                 if clean_json.endswith("```"):
                     clean_json = clean_json[:-3]
            
            response_data = json.loads(clean_json)
            best_chunk_id = response_data.get("chunk_id")
            
            if best_chunk_id is not None and best_chunk_id in chunk_map:
                target_chunk = chunk_map[best_chunk_id]
                print("\n--- Match Found ---")
                print(f"File: {target_chunk['path']}")
                print(f"Content Snippet: {target_chunk['content'][:200]}...")
            else:
                print("LLM could not identify a relevant chunk.")
                
        except json.JSONDecodeError:
            print(f"Failed to parse LLM response: {output_text}")

    else:
        print(f"Directory '{data_dir}' not found.")

if __name__ == "__main__":
    main()
