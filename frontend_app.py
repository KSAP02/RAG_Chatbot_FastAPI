import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title = "AI Document Chatbot with FastAPI",
    page_icon = "🤖",
    layout = "wide"
)

st.title("Document Chatbot")

# Initializing Session State variables
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
@st.cache_data
def process_document(uploaded_file):
    
    st.write(f"{uploaded_file.name} uploaded successfully!")
    
    # This is a multipart-form request that FastAPI automatically converts into an UploadFile object.
    file = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
    # .getvalue =>gets the raw file content/ reads the entire file as bytes
    # .type gets the MIME type (e.g., application/pdf))
    
    # Send file to backend for processing and retrieve the processed documents
    
    # file is a dict here not an object of type UploadFile
    response = requests.post(f"{BACKEND_URL}/process_document/", files=file)
    
    if response.status_code == 200:
        processed_data = response.json()
        return processed_data # this is a dict with key as documents and value as a list of dicts
    
    else:
        st.error("Error processing document")
        return None

@st.cache_data
def get_chunks_metadatas(chunks: list):
    
    # print("Inside get_chunks_metadatas")
    # print(chunks[0]['page_content'])
    texts = [chunk['page_content'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]
    
    return texts, metadatas

@st.cache_data
def create_vector_store(chunks, metadatas):
    # Send chunks and metadatas to backend for vector store creation
    response = requests.post(f"{BACKEND_URL}/create_vector_store/", json={"chunks": chunks, "metadatas": metadatas})
    
    if response.status_code == 200:
        return response.json()
    else:
        return None
    
# Sidebar with document upload
with st.sidebar:
    st.header("Upload Documents")
    uploaded_file = st.file_uploader("Add documents to chat with",
                                     type=['pdf'])
    if uploaded_file:
        file_name = uploaded_file.name

        if 'processed_file' not in st.session_state or st.session_state.processed_file != file_name:
            with st.spinner("Processing Document..."):
                
                processed_data = process_document(uploaded_file)
                st.session_state.pd = processed_data
                # processed_data is of the form:
                    # {"documents":
                    # [{'page_content': '...', 'metadata': {'source': 'doc_0'}},  => chunk1
                    #  {'page_content': '...', 'metadata': {'source': 'doc_1'}}]   => chunk2
                    # }
                if processed_data:
                    
                    # st.session_state.documents contains list of dicts which
                    # are chunks of the document with content and metadata
                    st.session_state.documents = processed_data['documents']
                    
                    # get the page_content(chunks) and metadata and store them seperately
                    st.session_state.chunks, st.session_state.metadatas = get_chunks_metadatas(st.session_state.documents)
                    
                    # st.write(f"Received the chunks:{st.session_state.chunks}")
                    # st.write(f"Received the metadatas:{st.session_state.metadatas}")
                    
                    # send the chunks and metadata to the backend for vector store creation
                    st.session_state.response = create_vector_store(st.session_state.chunks, st.session_state.metadatas)
                    
                    # response stores a requests.Response object we can access the data by calling .json()
                    
                    st.session_state.processed_file = file_name

        # Handle response
        if st.session_state.response:
            st.success(f"{st.session_state.response['message']}!")
        else:
            st.error(f"Error: {st.session_state.response}")
        
        st.write(f"Document Processed Successfully and {st.session_state.response['message']}")
        
        # Display structured JSON Data
        st.subheader("Processed Document Data:")
        st.json(st.session_state.pd) # Shows nicely formatted JSON       
                    

# Dsiplay chat messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input("Ask me something...")

if user_input:
    
    # Add user message to chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
        })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
        
    # Get response from the backend
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # print(st.session_state.chunks)
            response = requests.post(f"{BACKEND_URL}/get_answer/",
                json={
                "query": user_input,
                "chunks": st.session_state.chunks
                })
            
            if response.status_code == 200:
                answer = response.json()
                st.markdown(answer)
                
                # Add assistant message to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })
            else:
                st.error("Error getting answer")