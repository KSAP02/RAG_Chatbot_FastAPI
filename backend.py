import os
import faiss
import requests
import numpy as np
from langchain.schema import Document
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from fastapi import FastAPI, UploadFile, File
import uvicorn
from dotenv import load_dotenv
from pydantic import BaseModel

# Initalizing Global Variables for easy access

LLM_MODEL = "google/flan-t5-base"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Get the Hugging Face API from .env file
load_dotenv()
api_key = os.getenv('HF_API_KEY')

llm_api_url = f"https://api-inference.huggingface.co/models/{LLM_MODEL}"

embedding_api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{EMBEDDING_MODEL}"

headers = {
            "Authorization": f"Bearer {api_key}",
            }

DOC_PROCESS_URL = "http://127.0.0.1:8001"

app = FastAPI()

# When FastAPI receives a file in multipart/form-data, it automatically wraps it in an UploadFile object.
# This means the file is not just raw bytes but an asynchronous file-like object.

# In the frontend, Streamlit’s file_uploader gives us raw bytes, which requests.post() can send easily.
# In the backend, FastAPI wraps files in an UploadFile object, which must be read first before sending via requests.post().
# Fix it by extracting the file content manually using await file.read() and wrapping it in a proper files tuple.

@app.post('/process_document/')
async def process_document(file: UploadFile = File(...)):
    
    # Read the file contents 
    contents = await file.read() # reads the actual binary content from the upleaded file
    
    print(f"Received file: {file.filename}", flush=True)
    print(f"File Size: {len(contents)} bytes", flush =True)
    
    # Using {"file": (file.filename, file.file, file.content_type)} ensures proper file handling across API calls.
    # Whenever sending files through requests.post(), always include filename and MIME type for compatibility.
    # This ensures your backend properly forwards the file to the document processor without loss of metadata.
    
    response = requests.post(f"{DOC_PROCESS_URL}/document_processing/", files={"file": (file.filename, contents, file.content_type)})
    
    # file.file → Sends the actual file as a stream, allowing efficient handling.
    # When making an API call using requests.post(), it expects files to be passed 
    # in a specific format as a multipart/form-data request.

    
    # Even though the response is already JSON, requests.post(...) in Python treats the response as raw text.
    # Calling .json() deserializes this JSON into a Python dictionary for easier access.   
    # print(f"Response: {response.json()}")
    processed_data = response.json()
    # for chunk in processed_data['documents']:
    #     print(chunk, flush=True)
    documents_list = processed_data['documents']
    # print(documents_list)
    # When you call response.json() on a FastAPI response, it converts the
    # JSON response into a Python dictionary.
    return processed_data

def get_embeddings(chunks: list[str]):
        print(f"Inside get_embeddings: {len(chunks)} chunks", flush=True)
        response = requests.post(embedding_api_url, headers=headers, json={"inputs":chunks}) 
        
        
        if response.status_code == 200:
            embeddings = response.json()
        else:
            raise Exception(f"Error fetching embeddings: {response.json()}")
        
        return embeddings

# FastAPI does not maintain state between requests, 
# so you need a global storage mechanism for storing the vector database

class VectorDatabase:
    def __init__(self):
        self.vector_store = None  # This will hold the FAISS index

    def create_vector_store(self, chunks: list[str], metadatas:list[dict]):
        """Creates and stores the FAISS vector database"""
        
        print(f"Inside create_vector_store: {len(chunks)} chunks", flush=True)
        print(chunks)
        embeddings = get_embeddings(chunks)
        dimension = len(embeddings[0])
        
        print(len(embeddings), flush=True)
        # Create FAISS index
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(np.array(embeddings, dtype=np.float32))

        # Create a document store
        documents_dict = {
            str(i): Document(page_content=chunks[i], metadata=metadatas[i])
            for i in range(len(chunks))
        }
        docstore = InMemoryDocstore(documents_dict)

        index_to_docstore_id = {i: str(i) for i in range(len(chunks))}

        # Store in global object
        self.vector_store = FAISS(
            index=faiss_index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            embedding_function=lambda x: self.get_embeddings(x),
        )

# Global vector database instance
vector_db = VectorDatabase()

# Endpoint to create the vector store
@app.post('/create_vector_store/')
async def create_vector_store(chunks: list[str], metadatas: list[dict]):
    #Creates and stores a FAISS vector database in the vector_db object instance
    vector_db.create_vector_store(chunks, metadatas)
    # print("Vector store created:", vector_db.vector_store is not None)
    # print("Vector store object:", vector_db.vector_store)

    return {"message": "Vector store created successfully"}

def retrieve_context(query, chunks, topk):
        query_embedding = get_embeddings([query])[0]
        
        print("Vector store created:", vector_db.vector_store is not None)
        print("Vector store object:", vector_db.vector_store)
        distances, indices = vector_db.vector_store.index.search(np.array([query_embedding], dtype=np.float32), k=topk)
        
        # indices[0] is used and not just indices as an array because indices is a 2d array
        retrieved_texts = [chunks[i] for i in indices[0] if i < len(chunks)]
        
        return "\n\n".join(retrieved_texts) # Return relevant strings as a one complete string

def get_response(prompt: str):
    payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": 512,
                "min_length": 30,
                "temperature": 0.5,
                "num_return_sequences": 1
            }
        }
    
    response = requests.post(llm_api_url, headers=headers, json=payload)
    print(f"In get_response: {response.json()}", flush=True)
    if response.status_code == 200:
        
        try:
            response_json = response.json()
            return (response_json)
        
        except Exception as e:
                print(f"Error processing response :{e}")
                print(f"Raw Response: {response.text}")
                return f"Error: {str(e)}"
    else:
        error_msg = f"Error ({response.status_code}): {response.text}"
        # print(error_msg)
        return error_msg
    
# Using pydantic model to parse JSON properly
class QueryRequest(BaseModel):
    query: str
    chunks: list[str]
    
# Endpoint to get the answer for a query
@app.post('/get_answer/')
# "request: QueryRequest" will tell fastapi to parse the json body correctly when sent from the frontend    
async def get_answer(request: QueryRequest): 
    
    print("Inside get_answer", flush=True)
    # "context" contains relevant text data to the query, from the vector database
    context = retrieve_context(request.query, request.chunks, topk=3)
    
    # Prompt Engineering
    prompt = f"""Based on the following information, please answer the question throughly.
    INFORMATION:
    {context}
    
    QUESTION:
    {request.query}
    
    """
    
    # get the response
    response = get_response(prompt)
    return response[0]['generated_text']



if __name__ == "__main__":
    uvicorn.run(app, host ='127.0.0.1', port=8000)
    # Run using : "uvicorn backend:app --host 127.0.0.1 --port 8000 --reload"