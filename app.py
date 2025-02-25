import os
import uuid
import tempfile
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import weaviate
from langchain_community.embeddings import OpenAIEmbeddings
# from langchain_community.vectorstores import Weaviate
from langchain_weaviate.vectorstores import WeaviateVectorStore as Weaviate
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    JSONLoader,
)
from langchain.chains import RetrievalQA
import dotenv
import json
from weaviate.classes.init import Auth
import weaviate.classes as wvc
from weaviate.classes.query import Filter

# Load environment variables from .env file
dotenv.load_dotenv()

# Constants
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
COLLECTION_NAME = "RAG_Documents"

# Initialize FastAPI app
app = FastAPI(title="RAG System API", description="API for RAG system using LangChain and Weaviate")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create Weaviate client
weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
)
print(weaviate_client.is_ready())

# Create schema if it doesn't exist
def initialize_weaviate_schema():
    """Initialize Weaviate schema if it doesn't exist"""
    if not weaviate_client.collections.exists(COLLECTION_NAME):
        weaviate_client.collections.create(
            name=COLLECTION_NAME,
            properties=[
                wvc.config.Property(
                    name="content",
                    data_type=wvc.config.DataType.TEXT
                ),
                wvc.config.Property(
                    name="filename",
                    data_type=wvc.config.DataType.TEXT
                ),
                wvc.config.Property(
                    name="doc_id",
                    data_type=wvc.config.DataType.TEXT
                ),
                wvc.config.Property(
                    name="chunk_id",
                    data_type=wvc.config.DataType.TEXT
                ),
                wvc.config.Property(
                    name="source",
                    data_type=wvc.config.DataType.TEXT
                ),
            ],
            # vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(),
            # vector_index_config=wvc.config.Configure.VectorIndex.hnsw(  # Or `flat` or `dynamic`
            #     distance_metric=wvc.config.VectorDistances.COSINE,
            #     quantizer=wvc.config.Configure.VectorIndex.Quantizer.bq(),
            # ),
        )
        # class_obj = {
        #     "class": COLLECTION_NAME,
        #     "vectorizer": "none",  # We'll provide our own vectors
        #     "properties": [
        #         {
        #             "name": "content",
        #             "dataType": ["text"],
        #         },
        #         {
        #             "name": "filename",
        #             "dataType": ["string"],
        #         },
        #         {
        #             "name": "doc_id",
        #             "dataType": ["string"],
        #         },
        #         {
        #             "name": "chunk_id",
        #             "dataType": ["string"],
        #         },
        #         {
        #             "name": "source",
        #             "dataType": ["string"],
        #         },
        #     ],
        # }
        # weaviate_client.collections.create(class_obj)
        print(f"Created schema for {COLLECTION_NAME}")

# Initialize schema on startup
@app.on_event("startup")
async def startup_event():
    initialize_weaviate_schema()

# Initialize embeddings and language model
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(
    model_name="gpt-4o-mini", 
    temperature=0, 
    openai_api_key=OPENAI_API_KEY
)

# Load documents based on file type
def load_document(file_path: str, file_type: str, filename: str):
    """Load document based on file type"""
    if file_type == "pdf":
        loader = PyPDFLoader(file_path)
        documents = loader.load()
    elif file_type == "docx":
        loader = Docx2txtLoader(file_path)
        documents = loader.load()
    elif file_type == "txt":
        loader = TextLoader(file_path)
        documents = loader.load()
    elif file_type == "json":
        # Using a simple JSON loader with jq-like selector
        loader = JSONLoader(
            file_path=file_path,
            jq_schema=".",
            text_content=False
        )
        documents = loader.load()
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    # Add metadata
    for doc in documents:
        doc.metadata["filename"] = filename
        doc.metadata["source"] = file_path
    
    return documents

# Split documents into chunks
def split_documents(documents):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    
    # Add chunk_id to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = str(i)
    
    return chunks

# Store documents in Weaviate
def store_documents(chunks, doc_id):
    """Store document chunks in Weaviate with doc_id and update existing if needed"""
    # Check if documents with the same filename already exist
    filename = chunks[0].metadata["filename"]
    
    # Delete existing documents with the same doc_id if they exist
    delete_documents_by_doc_id(doc_id)
    
    # Add doc_id to metadata for all chunks
    for chunk in chunks:
        chunk.metadata["doc_id"] = doc_id
    
    # Store documents in Weaviate
    vectorstore = Weaviate(
        client=weaviate_client,
        index_name=COLLECTION_NAME,
        text_key="content",
        embedding=embeddings,
        attributes=["filename", "doc_id", "chunk_id", "source"]
    )
    
    # Add documents with their metadata
    vectorstore.add_documents(chunks)
    
    return {"message": f"Successfully stored {len(chunks)} chunks for document {filename}"}

# Delete documents by doc_id
def delete_documents_by_doc_id(doc_id):
    """Delete documents with the given doc_id from Weaviate"""
    collection = weaviate_client.collections.get(COLLECTION_NAME)
    collection.data.delete_many(
        where=Filter.by_property("doc_id").equal(doc_id)
    )
    print("Deleted successfully:", doc_id)

# Process document upload
def process_document(file_path, file_type, filename):
    """Process document upload and store in Weaviate"""
    # Generate a deterministic doc_id based on filename
    doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, filename))
    
    # Load document
    documents = load_document(file_path, file_type, filename)
    
    # Split into chunks
    chunks = split_documents(documents)
    
    # Store in Weaviate
    result = store_documents(chunks, doc_id)
    
    return {
        "doc_id": doc_id,
        "filename": filename,
        "chunks_count": len(chunks),
        "message": result["message"]
    }

# Get list of documents
def get_documents_list():
    """Get list of all stored documents"""
    documents = weaviate_client.collections.get(COLLECTION_NAME)
    query = documents.query.fetch_objects(
        return_properties=["filename", "doc_id"]
    )

    unique_docs = {}
    
    for obj in query.objects:
        doc_id = obj.properties.get("doc_id")
        if doc_id and doc_id not in unique_docs:
            unique_docs[doc_id] = {
                "doc_id": doc_id,
                "filename": obj.properties.get("filename"),
                "id": obj.uuid
            }
    
    return list(unique_docs.values())

# Query function
def query_documents(question, doc_ids=None):
    """Query documents based on question and optional doc_ids filter"""
    
    # Create the vectorstore with the Weaviate client
    vectorstore = Weaviate(
        client=weaviate_client,
        index_name=COLLECTION_NAME,
        text_key="content",
        embedding=embeddings,
        attributes=["filename", "doc_id", "chunk_id", "source", "content"]
    )

        
    # Create a filter if doc_ids are provided
    if doc_ids and len(doc_ids) > 0:
        search_filter = Filter.by_property("doc_id").equal(doc_ids[0])
    
    # Create a retriever with the metadata filter
    retriever = vectorstore.as_retriever(
        # search_type="mmr",
        search_kwargs={
            "k": 3,  # Number of documents to retrieve
            "filters": search_filter
        }
    )

    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    
    # Run the query
    result = qa_chain.invoke({"query": question})
    
    # Format the result with source documents and their metadata
    response = {
        "answer": result["result"],
        "source_documents": []
    }
    
    # Include source document information
    for doc in result["source_documents"]:
        source_doc = {
            "content": doc.page_content,
            "filename": doc.metadata.get("filename", ""),
            "doc_id": doc.metadata.get("doc_id", ""),
            "chunk_id": doc.metadata.get("chunk_id", ""),
            "source": doc.metadata.get("source", "")
        }
        response["source_documents"].append(source_doc)
    
    return response

# API Models
class DocumentResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_count: int
    message: str

class DocumentList(BaseModel):
    documents: List[Dict[str, Any]]

class QueryRequest(BaseModel):
    question: str
    doc_ids: Optional[List[str]] = None

class QueryResponse(BaseModel):
    answer: str
    source_documents: List[Dict[str, Any]]

# API Routes
@app.post("/upload", response_model=DocumentResponse)
async def upload_document(
    file: UploadFile = File(...),
):
    """Upload and process a document"""
    # Check file extension
    filename = file.filename
    file_extension = filename.split(".")[-1].lower()
    
    if file_extension not in ["pdf", "docx", "txt", "json"]:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_extension}. Supported types are PDF, DOCX, TXT, and JSON."
        )
    
    # Save file to temp directory
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as temp_file:
        temp_file.write(await file.read())
        temp_file_path = temp_file.name
    
    try:
        # Process the document
        result = process_document(temp_file_path, file_extension, filename)
        return result
    finally:
        # Clean up the temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.get("/documents", response_model=DocumentList)
async def list_documents():
    """List all stored documents"""
    documents = get_documents_list()
    return {"documents": documents}

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query the documents based on a question and optional document filters"""
    result = query_documents(request.question, request.doc_ids)
    return result

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document by doc_id"""
    delete_documents_by_doc_id(doc_id)
    return {"message": f"Document with doc_id {doc_id} has been deleted"}

@app.get("/")
async def read_root():
    return {"message": "Welcome to the RAG System API"}

# Main function to run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8005)