# Multi-Format RAG (Retrieval-Augmented Generation) System
# Supports: PDF, DOCX, TXT, CSV, XLSX, JSON, HTML, MD

import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

# Document Processing
from PyPDF2 import PdfReader
import docx
import pandas as pd
import json
from bs4 import BeautifulSoup
import markdown

# Embeddings & Vector Store
from sentence_transformers import SentenceTransformer
import faiss
import pickle

# LLM Integration
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch



class DocumentLoader:
    """Load and process documents from multiple file formats"""
    
    def __init__(self):
        self.supported_formats = {
            '.pdf': self._load_pdf,
            '.docx': self._load_docx,
            '.txt': self._load_txt,
            '.csv': self._load_csv,
            '.xlsx': self._load_excel,
            '.json': self._load_json,
            '.html': self._load_html,
            '.md': self._load_markdown
        }
    
    def load_document(self, file_path: str) -> Dict[str, Any]:
        """Load a document and return its content with metadata"""
        
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        content = self.supported_formats[file_extension](file_path)
        
        return {
            'content': content,
            'filename': file_path.name,
            'file_type': file_extension,
            'file_path': str(file_path)
        }
    
    def _load_pdf(self, file_path: Path) -> str:
        """Extract text from PDF"""
        reader = PdfReader(str(file_path))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    
    def _load_docx(self, file_path: Path) -> str:
        """Extract text from DOCX"""
        doc = docx.Document(str(file_path))
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text
    
    def _load_txt(self, file_path: Path) -> str:
        """Load plain text file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _load_csv(self, file_path: Path) -> str:
        """Load CSV and convert to text"""
        df = pd.read_csv(file_path)
        # Convert to readable text format
        text = f"CSV Data from {file_path.name}:\n\n"
        text += df.to_string(index=False)
        return text
    
    def _load_excel(self, file_path: Path) -> str:
        """Load Excel file and convert to text"""
        xl_file = pd.ExcelFile(file_path)
        text = f"Excel Data from {file_path.name}:\n\n"
        
        for sheet_name in xl_file.sheet_names:
            df = pd.read_excel(xl_file, sheet_name=sheet_name)
            text += f"\n--- Sheet: {sheet_name} ---\n"
            text += df.to_string(index=False) + "\n"
        
        return text
    
    def _load_json(self, file_path: Path) -> str:
        """Load JSON and convert to readable text"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert JSON to readable format
        return json.dumps(data, indent=2)
    
    def _load_html(self, file_path: Path) -> str:
        """Extract text from HTML"""
        with open(file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        return soup.get_text(separator='\n', strip=True)
    
    def _load_markdown(self, file_path: Path) -> str:
        """Load Markdown file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            md_content = f.read()
        
        # Convert to HTML then extract text for better formatting
        html = markdown.markdown(md_content)
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text(separator='\n', strip=True)


class TextChunker:
    """Split documents into manageable chunks for embedding"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str, metadata: Dict) -> List[Dict]:
        """Split text into overlapping chunks"""
        
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.overlap):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append({
                'text': chunk_text,
                'metadata': {
                    **metadata,
                    'chunk_id': len(chunks),
                    'start_word': i,
                    'end_word': i + len(chunk_words)
                }
            })
        
        return chunks


class VectorStore:
    """Store and retrieve document embeddings using FAISS"""
    
    def __init__(self, embedding_model: str = 'all-MiniLM-L6-v2'):
        self.encoder = SentenceTransformer(embedding_model)
        self.dimension = self.encoder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.chunks = []
    
    def add_documents(self, chunks: List[Dict]):
        """Add document chunks to the vector store"""
        
        texts = [chunk['text'] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        
        # Add to FAISS index
        self.index.add(np.array(embeddings).astype('float32'))
        
        # Store chunks for retrieval
        self.chunks.extend(chunks)
        
        print(f"Added {len(chunks)} chunks to vector store")
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant chunks"""
        
        # Encode query
        query_embedding = self.encoder.encode([query])
        
        # Search FAISS index
        distances, indices = self.index.search(
            np.array(query_embedding).astype('float32'), 
            k
        )
        
        # Retrieve chunks
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                results.append({
                    'chunk': self.chunks[idx],
                    'score': float(distance)
                })
        
        return results
    
    def save(self, path: str):
        """Save vector store to disk"""
        faiss.write_index(self.index, f"{path}_index.faiss")
        with open(f"{path}_chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        print(f"Vector store saved to {path}")
    
    def load(self, path: str):
        """Load vector store from disk"""
        self.index = faiss.read_index(f"{path}_index.faiss")
        with open(f"{path}_chunks.pkl", 'rb') as f:
            self.chunks = pickle.load(f)
        print(f"Vector store loaded from {path}")


class RAGModel:
    """Complete RAG system combining retrieval and generation"""
    
    def __init__(
        self,
        model_name: str = "microsoft/DialoGPT-medium",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        # Initialize components
        self.document_loader = DocumentLoader()
        self.chunker = TextChunker(chunk_size=512, overlap=50)
        self.vector_store = VectorStore(embedding_model)
        
        # Load LLM
        print(f"Loading language model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        
        print(f"RAG Model initialized on {self.device}")
    
    def ingest_documents(self, file_paths: List[str]):
        """Ingest multiple documents into the RAG system"""
        
        print(f"\nIngesting {len(file_paths)} documents...")
        
        all_chunks = []
        
        for file_path in file_paths:
            try:
                # Load document
                doc = self.document_loader.load_document(file_path)
                print(f"✓ Loaded: {doc['filename']}")
                
                # Chunk document
                chunks = self.chunker.chunk_text(doc['content'], {
                    'filename': doc['filename'],
                    'file_type': doc['file_type']
                })
                
                all_chunks.extend(chunks)
                
            except Exception as e:
                print(f"✗ Error loading {file_path}: {str(e)}")
        
        # Add all chunks to vector store
        if all_chunks:
            self.vector_store.add_documents(all_chunks)
            print(f"\n✓ Successfully ingested {len(all_chunks)} chunks")
        else:
            print("\n✗ No chunks to ingest")
    
    def query(
        self,
        question: str,
        k: int = 3,
        max_length: int = 200,
        temperature: float = 0.7
    ) -> Dict[str, Any]:
        """Query the RAG system"""
        
        # Retrieve relevant chunks
        print(f"\nSearching for relevant context...")
        results = self.vector_store.search(question, k=k)
        
        if not results:
            return {
                'answer': "No relevant information found in the documents.",
                'sources': []
            }
        
        # Prepare context from retrieved chunks
        context = "\n\n".join([
            f"[Source: {r['chunk']['metadata']['filename']}]\n{r['chunk']['text']}"
            for r in results
        ])
        
        # Create prompt
        prompt = f"""Context information:
{context}

Question: {question}

Answer based on the context above:"""
        
        # Generate answer
        print("Generating answer...")
        inputs = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_length=max_length + inputs.shape[1],
                temperature=temperature,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        answer = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        # Prepare sources
        sources = [
            {
                'filename': r['chunk']['metadata']['filename'],
                'file_type': r['chunk']['metadata']['file_type'],
                'chunk_id': r['chunk']['metadata']['chunk_id'],
                'relevance_score': r['score'],
                'text_preview': r['chunk']['text'][:200] + "..."
            }
            for r in results
        ]
        
        return {
            'answer': answer.strip(),
            'sources': sources,
            'context': context
        }
    
    def save_vector_store(self, path: str = "./rag_vector_store"):
        """Save the vector store"""
        self.vector_store.save(path)
    
    def load_vector_store(self, path: str = "./rag_vector_store"):
        """Load a previously saved vector store"""
        self.vector_store.load(path)


# Example Usage
if __name__ == "__main__":
    
    # Initialize RAG model
    rag = RAGModel(
        model_name="microsoft/DialoGPT-medium",
        embedding_model="all-MiniLM-L6-v2"
    )
    
    # Example: Ingest documents from various formats
    documents = [
        "./documents/realistic_restaurant_reviews.csv"
    ]
    
    # Ingest documents
    rag.ingest_documents(documents)
    
    # Save vector store for future use
    rag.save_vector_store("./my_rag_store")
    
    # Query the system
    question = "What are the key findings in the report?"
    result = rag.query(question, k=5)
    
    print("\n" + "="*80)
    print("QUESTION:", question)
    print("="*80)
    print("\nANSWER:")
    print(result['answer'])
    print("\n" + "-"*80)
    print("SOURCES:")
    for i, source in enumerate(result['sources'], 1):
        print(f"\n{i}. {source['filename']} ({source['file_type']})")
        print(f"   Relevance: {source['relevance_score']:.4f}")
        print(f"   Preview: {source['text_preview']}")
    print("="*80)
    
    # Example: Load existing vector store
    rag.load_vector_store("./my_rag_store")
    
    # Interactive mode
    print("\n\n🤖 RAG System Ready! Type 'quit' to exit.")
    while True:
        user_query = input("\n💬 Your question: ")
        if user_query.lower() in ['quit', 'exit', 'q']:
            break
        
        result = rag.query(user_query)
        print(f"\n🤖 Answer: {result['answer']}")
        print(f"\n📚 Sources: {len(result['sources'])} relevant chunks found")


# file_path = "./documents/realistic_restaurant_reviews.csv"
# document_loader = DocumentLoader()
# doc = document_loader.load_document(file_path)
# print(f"✓ Loaded: {doc['filename']}")
# print(len(doc))