import asyncio
import io
import os
from typing import List, Dict, Optional
import sqlite3
import json
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain.output_parsers import PydanticOutputParser
import faiss
import requests
import pdfplumber
import re

# Initialize FastAPI app
app = FastAPI()

# Initialize SQLite database and models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
def get_chat_history(session_id):
    return SQLChatMessageHistory(session_id, "sqlite:///chat_history.db")

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

# Create a new table for embeddings metadata
cursor.execute("""
    CREATE TABLE IF NOT EXISTS embeddings_metadata (
        id TEXT PRIMARY KEY,  -- video_id or doc_id
        type TEXT NOT NULL,   -- 'video' or 'pdf'
        transcript TEXT NOT NULL,
        embedding BLOB        -- Store embeddings as binary data
    )
""")
conn.commit()

# Create a new table for chat history
cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT NOT NULL,
        message TEXT NOT NULL,
        role TEXT NOT NULL,  
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
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
docstore = InMemoryDocstore({})
index_to_docstore_id = {}

# Load FAISS index if it exists
FAISS_INDEX_PATH = "faiss_index"

def save_faiss_index(vector_store):
    """Save the FAISS index to disk."""
    vector_store.save_local(FAISS_INDEX_PATH)

def load_faiss_index():
    """Load the FAISS index from disk."""
    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(FAISS_INDEX_PATH, embedding_model.encode, allow_dangerous_deserialization=True)
    return None
# Load FAISS index if it exists
vector_store = load_faiss_index()
if vector_store is None:
    # Create a new FAISS index if it doesn't exist
    vector_store = FAISS(
        embedding_function=embedding_model.encode,
        index=faiss.IndexFlatL2(dimension),
        docstore=InMemoryDocstore({}),
        index_to_docstore_id={}
    )
# Define request models
class VideoRequest(BaseModel):
    video_url: str
    create_test: bool = False
    session_id: str

class QuestionRequest(BaseModel):
    query: str
    session_id: str 

class PdfRequest(BaseModel):
    pdf_url: str
    session_id: str

class MCQRequest(BaseModel):
    url: str 
    session_id: str  # Can be a YouTube URL or a PDF URL

# Define the MCQ response model
class MCQResponse(BaseModel):
    question: str
    options: List[str]
    answer: str
    context: Optional[str] = None

# Define the MCQ model
class MCQ(BaseModel):
    """MCQ"""
    question: str = Field(description="The question from the text")
    options: List[str] = Field(description="This is options for the question")
    answer: int = Field(description="This is correct option number for the question")
    context: Optional[str] = Field(description="This is reference for the question")

# Initialize the parser
mcq_parser = PydanticOutputParser(pydantic_object=MCQ)

# Define the prompt template
mcq_prompt = PromptTemplate(
    template='''
    Create a multiple-choice question based on the following text. Ensure that the question is relevant and that the correct answer is one of the provided options.

    Text: {text}

    Please format your answer as a JSON object with the following keys:
    - question: The generated question.
    - options: A list of options.
    - answer: The index (0-based) of the correct option.
    
    Answer:
    ''',
    input_variables=['text'],
    partial_variables={'format_instructions': mcq_parser.get_format_instructions()}
)

mcq_chain = mcq_prompt | llm

# Setup QA chain
qa_chain = (
    {"context": lambda x: x["context"], "question": lambda x: x["question"]}
    | prompt
    | llm
)

# Helper functions
def embeddings_exist(id: str) -> bool:
    """Check if embeddings exist for given ID"""
    cursor.execute("SELECT 1 FROM embeddings_metadata WHERE id = ?", (id,))
    return cursor.fetchone() is not None

def store_embeddings_metadata(id: str, type: str, transcript: str, embedding: np.ndarray):
    """Store embeddings in metadata table"""
    cursor.execute(
        "INSERT OR REPLACE INTO embeddings_metadata (id, type, transcript, embedding) VALUES (?, ?, ?, ?)",
        (id, type, transcript, embedding.tobytes())
    )
    conn.commit()

def retrieve_embeddings(id: str) -> Optional[np.ndarray]:
    """Retrieve stored embeddings"""
    cursor.execute("SELECT embedding FROM embeddings_metadata WHERE id = ?", (id,))
    result = cursor.fetchone()
    return np.frombuffer(result[0], dtype=np.float32) if result else None

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

async def store_embeddings(id: str, transcript: str, is_video: bool = True):
    """Store transcript embeddings with metadata check"""
    try:
        if embeddings_exist(id):
            return  # Skip if already exists
            
        # Generate embeddings for the transcript
        embedding = embedding_model.encode(transcript)
        
        # Store embeddings in the metadata table
        store_embeddings_metadata(id, 'video' if is_video else 'pdf', transcript, embedding)

        # Split transcript into meaningful chunks
        chunks = [s.strip() for s in re.split(r'[.!?]+', transcript) if s.strip()]
        if not chunks:
            raise ValueError("No valid chunks extracted from transcript")

        # Add chunks and embeddings to FAISS
        vector_store.add_texts(
            texts=chunks,
            metadatas=[{"source": 'video' if is_video else 'pdf'} for _ in chunks]
        )
        
        # Save the FAISS index to disk
        save_faiss_index(vector_store)
        
        print(f"Added {len(chunks)} chunks to FAISS index.")  # Debug log
        
        return len(chunks)
        
    except Exception as e:
        raise HTTPException(500, f"Error storing embeddings: {str(e)}")

@app.post("/generateMCQ", response_model=List[MCQResponse])
async def generate_mcq(request: MCQRequest):
    try:
        # Retrieve chat history for the session
        chat_history = get_chat_history(request.session_id)

        # Step 1: Extract text based on the URL type
        if "youtube.com" in request.url or "youtu.be" in request.url:
            # Extract transcript from YouTube video
            video_id = get_video_id(request.url)
            text = await get_transcript(video_id)
        else:
            # Extract text from PDF
            response = requests.get(request.url)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to download the PDF. Please check the URL."
                )
            pdf_bytes = io.BytesIO(response.content)
            text = ""
            with pdfplumber.open(pdf_bytes) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"

        if not text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the provided URL."
            )

        # Step 2: Generate MCQs using the llama3.2:3b model
        mcqs = await generate_mcq_from_text(text)

        # Add user message and assistant response to chat history
        chat_history.add_user_message(f"Generated MCQs from URL: {request.url}")
        chat_history.add_ai_message(json.dumps(mcqs))

        return mcqs

    except HTTPException as he:
        raise he
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating MCQs from text: {str(e)}")

async def generate_mcq_from_text(text: str) -> List[Dict]:
    """
    Generate MCQs from the given text using the llama3.2:3b model.
    """
    try:
        # Retrieve relevant context using vector embeddings
        docs = vector_store.similarity_search(text, k=5)
       # Debug log
        
        if not docs:
            print("No relevant documents found in FAISS index.")  # Debug log
            return []

        mcqs = []
        for doc in docs:
            try:
                response = await mcq_chain.ainvoke({'text': doc.page_content})

                # Extract JSON from the model's response
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if not json_match:
                    print(f"No JSON found in response: {response.content}")
                    continue

                # Parse the JSON object
                mcq = json.loads(json_match.group(0))

                # Validate that required fields are present
                if all(key in mcq for key in ["question", "options", "answer"]):
                    mcq["answer"] = str(mcq["answer"])  # Convert answer index to string
                    mcq["context"] = doc.page_content  # Add context from the retrieved document
                    mcqs.append(mcq)
                else:
                    print(f"Invalid MCQ format: {response.content}")
            except json.JSONDecodeError as e:
                print(f"Error parsing MCQ response: {str(e)}")
                print(f"Response content: {response.content}")
                continue

        return mcqs

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating MCQs from text: {str(e)}"
        )

@app.post("/chatPdf")
async def process_Pdf(request: PdfRequest):
    try:
        # Retrieve chat history for the session
        chat_history = get_chat_history(request.session_id)
        pdf_id = request.pdf_url

        if embeddings_exist(pdf_id):
            chat_history.add_user_message(f"Processed PDF: {pdf_id}")
            chat_history.add_ai_message("PDF already processed")
            return {"pdf_id": pdf_id, "status": "cached"}

        # Step 1: Download the PDF from the URL
        response = requests.get(request.pdf_url)
        if response.status_code != 200:
            raise HTTPException(
                status_code=400,
                detail="Failed to download the PDF. Please check the URL."
            )

        # Step 2: Wrap the PDF content in a BytesIO object
        pdf_bytes = io.BytesIO(response.content)

        # Step 3: Extract text from the PDF
        pdf_text = ""
        with pdfplumber.open(pdf_bytes) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:  # Ensure text is extracted
                    pdf_text += page_text + "\n"

        if not pdf_text.strip():
            raise HTTPException(
                status_code=400,
                detail="No text could be extracted from the PDF."
            )

        # Step 4: Store embeddings
        chunks_processed = await store_embeddings(pdf_id, pdf_text, is_video=False)

        # Add context to chat history
        chat_history.add_user_message(f"Processed PDF: {request.pdf_url}")
        chat_history.add_ai_message(f"PDF processed. {chunks_processed} chunks stored.")

        return {
            "pdf_id": pdf_id,
            "message": f"PDF processed. {chunks_processed} chunks stored.",
            "status": "success"
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing PDF: {str(e)}"
        )

@app.post("/chatYoutube")
async def process_video(request: VideoRequest):
    try:
        # Retrieve chat history for the session
        chat_history = get_chat_history(request.session_id)

        video_id = get_video_id(request.video_url)
        
        # Check if already processed
        if embeddings_exist(video_id):
            chat_history.add_user_message(f"Processed YouTube video: {request.video_url}")
            chat_history.add_ai_message("Transcript already processed and stored.")
            return {
                "video_id": video_id,
                "message": "Transcript already processed and stored.",
                "status": "cached"
            }
        
        # Process new video
        transcript = await get_transcript(video_id)
        chunks_processed = await store_embeddings(video_id, transcript)
        
        # Add context to chat history
        chat_history.add_user_message(f"Processed YouTube video: {request.video_url}")
        chat_history.add_ai_message(f"Transcript processed. {chunks_processed} chunks stored.")

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
        # Retrieve chat history for the session
        chat_history = get_chat_history(request.session_id)

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
        
        # Add user message and assistant response to chat history
        chat_history.add_user_message(request.query)
        chat_history.add_ai_message(response.content.strip())
        
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