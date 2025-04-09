import io
import fitz
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from fastapi import FastAPI, HTTPException, UploadFile, File

app = FastAPI()

def chunk_text(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    chunks = text_splitter.split_documents([document])
    
    return [
        {"page_content": chunk.page_content, "metadata": chunk.metadata}
        for chunk in chunks
    ]

@app.post("/document_processing/")
async def document_processing(file: UploadFile = File(...)):
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Unsupported file type. Only PDF files are supported.")
    
    file_bytes = await file.read() # Read file as binary
    pdf_stream = io.BytesIO(file_bytes) # Convert bytes to file-like object

    doc = fitz.open("pdf", pdf_stream)
    
    text = "\n".join([page.get_text() for page in doc])
    
    document = Document(page_content=text, metadata={"source": file.filename})
    
    chunked_documents = chunk_text(document)

    print("Returning from document_processor:", chunked_documents)  # Debug print
    
    return {"documents": chunked_documents}  # Return the processed documents

# FastAPI automatically tries to serialize the response using Python’s built-in json module, 
# which only supports primitive data types like:
# dict, list
# str, int, float, bool, None
# But LangChain's Document class contains complex Python objects, which cannot be serialized without explicit conversion.
if __name__=="__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8001)
    # Run using : "uvicorn document_processing:app --host 127.0.0.1 --port 8001 --reload"