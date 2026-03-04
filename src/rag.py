import os
import json
import faiss
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import concurrent.futures
import pickle
from typing import List, Dict, Any

class RAGSystem:
    def __init__(self, index_path="embeddings"):
        print("Loading RAG model (SentenceTransformer)...")
        self.index_path = index_path
        os.makedirs(index_path, exist_ok=True)
        self.model = SentenceTransformer('all-MiniLM-L6-v2') 
        self.vector_dim = 384 
        self.chunks = [] 
        self.bm25 = None
        self.faiss_index = None

    def _process_file(self, file_info):
        path, content_lines = file_info
        file_chunks = []
        
        path_context = f"File Path: {path}"
        
        file_chunks.append({
            'path': path,
            'content': path 
        })
        
        file_chunks.append({
            'path': path,
            'content': path_context
        })

        for line in content_lines:
            line = line.strip()
            if line:
                combined_content = f"{path_context}\nContent: {line}"
                file_chunks.append({
                    'path': path,
                    'content': combined_content
                })
        return file_chunks

    def ingest(self, data_index: Dict[str, Any]):
        files_to_process = []
        
        def traverse(current_path, data):
            for key, value in data.items():
                if isinstance(value, dict):
                    traverse(os.path.join(current_path, key), value)
                elif isinstance(value, list):
                    files_to_process.append((os.path.join(current_path, key), value))
        
        traverse("", data_index)
        print(f"Total files to process: {len(files_to_process)}")
        flat_chunks = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = executor.map(self._process_file, files_to_process)
            for res in results:
                flat_chunks.extend(res)
        
        print(f"Total chunks created: {len(flat_chunks)}")

        for i, chunk in enumerate(flat_chunks):
            chunk['id'] = i
        self.chunks = flat_chunks
        
        if not self.chunks:
            print("No chunks found to index.")
            return

        corpus = [chunk['content'] for chunk in self.chunks]
        
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print("Generating embeddings...")
        embeddings = self.model.encode(corpus, batch_size=32, show_progress_bar=True)
        self.faiss_index = faiss.IndexFlatL2(self.vector_dim)
        self.faiss_index.add(np.array(embeddings).astype('float32'))
        
        self.save_index()
        print(f"Index built with {len(self.chunks)} chunks.")

    def save_index(self):
        df = pd.DataFrame(self.chunks)
        df.to_csv(os.path.join(self.index_path, 'chunks.csv'), index=False)
        faiss.write_index(self.faiss_index, os.path.join(self.index_path, 'vector.index'))
        with open(os.path.join(self.index_path, 'bm25.pkl'), 'wb') as f:
            pickle.dump(self.bm25, f)

    def load_index(self):
        if not os.path.exists(os.path.join(self.index_path, 'chunks.csv')):
            return False
            
        try:
            df = pd.read_csv(os.path.join(self.index_path, 'chunks.csv'))
            self.chunks = df.to_dict('records')
            
            self.faiss_index = faiss.read_index(os.path.join(self.index_path, 'vector.index'))
            
            with open(os.path.join(self.index_path, 'bm25.pkl'), 'rb') as f:
                self.bm25 = pickle.load(f)
            return True
        except Exception as e:
            print(f"Error loading index: {e}")
            return False

    def search(self, query: str, k=5, alpha=0.5):
        if not self.chunks:
            return []
            
        query_embedding = self.model.encode([query])
        D, I = self.faiss_index.search(np.array(query_embedding).astype('float32'), min(k * 5, len(self.chunks)))
        
        vector_scores = {}
        for idx, dist in zip(I[0], D[0]):
            if idx != -1:
                sim = 1 / (1 + dist) 
                vector_scores[idx] = sim

        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)

        if len(bm25_scores) > 0:
            max_bm25 = max(bm25_scores)
            if max_bm25 == 0: max_bm25 = 1
        else:
            max_bm25 = 1

        bm25_top_indices = np.argsort(bm25_scores)[::-1][:min(k * 5, len(self.chunks))]
        
        candidates = set(vector_scores.keys()) | set(bm25_top_indices)
        
        results = []
        for idx in candidates:
            idx = int(idx)
            v_score = vector_scores.get(idx, 0)
            b_score = bm25_scores[idx] / max_bm25
            
            final_score = (v_score * alpha) + (b_score * (1 - alpha))
            
            
            results.append({
                'chunk_id': self.chunks[idx]['id'],
                'path': self.chunks[idx]['path'],
                'content': self.chunks[idx]['content'],
                'score': final_score
            })
            
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]
