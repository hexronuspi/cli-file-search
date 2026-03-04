## Examples

![Screenshot 1](https://raw.githubusercontent.com/hexronuspi/cli-file-search/main/examples/Screenshot%202026-03-04%20055358.png)

![Screenshot 2](https://raw.githubusercontent.com/hexronuspi/cli-file-search/main/examples/Screenshot%202026-03-04%20055408.png)

![Screenshot 3](https://raw.githubusercontent.com/hexronuspi/cli-file-search/main/examples/Screenshot%202026-03-04%20060539.png)

![Screenshot 4](https://raw.githubusercontent.com/hexronuspi/cli-file-search/main/examples/Screenshot%202026-03-04%20060642.png)
____________________________________

## 1. Query Input
**User Query:**  
`q = "find the file where I have stored my SBI password"`

---

## 2. Ingestion (Preprocessing)
The system scans the `data/` directory and breaks down large files into smaller, manageable pieces (chunks).

**Example Chunking:**
- **File:** `data/password/sbi.txt`
- **Output Chunk:**
  ```json
  {
      "id": 42,
      "path": "data/password/sbi.txt",
      "content": "File Path: data/password/sbi.txt\nContent: sbi_password = 'secure_pass_123'"
  }
  ```

*Techniques Used:*
- **Text Embedding:** Converts chunk text to a vector (384-dim) using `SentenceTransformer`.
- **Keyword Indexing:** Builds a BM25 index for exact keyword matching (e.g., "SBI").

---

## 3. Storage
The processed data is stored for fast retrieval.

- **Vector Index (FAISS):** Stores the 384-dim vectors for semantic search.
- **Metadata (CSV):** Stores the actual text content and file paths mapped to chunk IDs.

---

## 4. Hybrid Search
The system retrieves the top candidate chunks using a weighted combination of two algorithms.

**Formula:**
`Score = (Vector_Similarity * 0.5) + (BM25_Score * 0.5)`

**Result (Top-K Chunks):**
1. **Chunk #42** (Score: 0.92) - `data/password/sbi.txt` ("...sbi_password...")
2. **Chunk #15** (Score: 0.45) - `data/story.txt` ("...bank of a river...")
3. **Chunk #88** (Score: 0.12) - `data/random.txt` ("...random text...")

---

## 5. LLM Ranking (The "Judge")
Instead of sending entire files, we send **only the Top-K chunks** to the LLM to identify the correct one.

**Prompt to LLM:**
```json
[
  {"id": 42, "source": "data/password/sbi.txt", "content": "...sbi_password..."},
  {"id": 15, "source": "data/story.txt", "content": "...bank..."}
]
```
**Task:** "Return the ID of the chunk that contains the answer."

---

## 6. Final Output
The LLM analyzes the content and selects the best match.

**LLM Response:**  
`{"chunk_id": 42}`

**System Output:**  
`File Found: data/password/sbi.txt`

