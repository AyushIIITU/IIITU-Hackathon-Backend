import asyncio
import sqlite3
import numpy as np
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
import faiss
import re

# Initialize FastAPI app
app = FastAPI()

# Initialize SQLite database and models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
conn = sqlite3.connect("embeddings.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS embeddings")
cursor.execute("""
    CREATE TABLE IF NOT EXISTS embeddings (
        video_id TEXT PRIMARY KEY,
        transcript TEXT
    )
""")
conn.commit()

# Define the prompt template
prompt_template = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Answer in bullet points. Make sure your answer is relevant to the question and it is answered from the context only.
    Question: {question} 
    Context: {context} 
    Answer:
"""

# Setup LangChain components
prompt = ChatPromptTemplate.from_template(prompt_template)
base_url = "http://localhost:11434"
model = 'llama3.2:3b'
llm = ChatOllama(model=model, base_url=base_url)

# Initialize FAISS components
dimension = 384  # Dimension for MiniLM
index = faiss.IndexFlatL2(dimension)
docstore = InMemoryDocstore({})  # Use InMemoryDocstore for storing documents
index_to_docstore_id = {}  # Use a simple dictionary for mapping indices to docstore IDs

vector_store = FAISS(
    embedding_function=embedding_model.encode,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)

# Define request models
class VideoRequest(BaseModel):
    video_url: str
    create_test: bool = False

class QuestionRequest(BaseModel):
    query: str

# Setup QA chain
qa_chain = (
    {"context": lambda x: x["context"], "question": lambda x: x["question"]}
    | prompt
    | llm
)

def get_video_id(url: str) -> str:
    try:
        if "youtube.com/watch?v=" in url:
            return url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            return url.split("youtu.be/")[1].split("?")[0]
        raise ValueError("Invalid YouTube URL format")
    except Exception as e:
        raise ValueError(f"Could not extract video ID: {str(e)}")

async def get_transcript(video_id: str) -> str:
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([t["text"] for t in transcript])
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not fetch transcript for video {video_id}: {str(e)}"
        )

def fetch_cached_transcript(video_id: str) -> str:
    try:
        cursor.execute(
            "SELECT transcript FROM embeddings WHERE video_id = ?",
            (video_id,)
        )
        result = cursor.fetchone()
        return result[0] if result else None
    except sqlite3.Error as e:
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )

async def store_embeddings(video_id: str, transcript: str):
    """Store transcript embeddings in FAISS and transcript in SQLite."""
    try:
        # Split transcript into chunks
        chunks = [s.strip() for s in re.split(r'[.!?]+', transcript) if s.strip()]
        if not chunks:
            raise ValueError("No valid chunks extracted from transcript")
        
        # Add chunks to FAISS
        vector_store.add_texts(
            texts=chunks,
            metadatas=[{"video_id": video_id} for _ in chunks]
        )
        
        # Store transcript in SQLite
        cursor.execute(
            "INSERT OR REPLACE INTO embeddings (video_id, transcript) VALUES (?, ?)",
            (video_id, transcript)
        )
        conn.commit()
        
        return len(chunks)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error storing embeddings: {str(e)}"
        )

@app.post("/chat")
async def process_video(request: VideoRequest):
    try:
        video_id = get_video_id(request.video_url)
        
        # Check if already processed
        existing_transcript = fetch_cached_transcript(video_id)
        if existing_transcript:
            return {
                "video_id": video_id,
                "message": "Transcript already processed and stored.",
                "status": "cached"
            }
        
        # Process new video
        transcript = await get_transcript(video_id)
        chunks_processed = await store_embeddings(video_id, transcript)
        
        return {
            "video_id": video_id,
            "message": f"Transcript processed. {chunks_processed} chunks stored.",
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask_question(request: QuestionRequest):
    try:
        # Retrieve relevant context from FAISS
        docs = vector_store.similarity_search(request.query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        if not context:
            return {
                "response": "No relevant information found to answer this question.",
                "status": "success"
            }
        
        # Generate answer using QA chain
        response = qa_chain.invoke({"context": context, "question": request.query})
        
        return {
            "response": response.content.strip(),
            "status": "success",
            "context": [doc.page_content for doc in docs]
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing question: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)