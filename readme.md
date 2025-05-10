**Technologies Used**: Streamlit, LangChain, FAISS, Hugging Face Transformers, FastAPI, Python
This project focused on building a Retrieval-Augmented Generation (RAG) chatbot to answer questions based on uploaded documents. The system was designed to assist users in querying large documents through natural language, without manually scanning content.

**Key Components and Contributions:**
- Developed a full RAG pipeline using LangChain, integrating document and semantic chunking, semantic embedding, vector store creation, and LLM inference.
- Implemented document embedding and retrieval using FAISS for fast similarity search.
- Designed a preprocessing module for semantic chunking to enhance retrieval accuracy.
- Integrated Hugging Face Transformers to generate context-aware answers based on retrieved chunks.
- Developed a user-friendly Streamlit interface for document upload and chatbot interaction.
- Built modular FastAPI endpoints for backend processing and model interaction.

**Outcome:**
The system successfully demonstrated how users can obtain concise and relevant answers from large documents through an interactive UI. It also served as a template for future document-based assistants.
