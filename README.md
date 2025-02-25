# RAG System with LangChain and Weaviate

This project implements a Retrieval-Augmented Generation (RAG) system using LangChain and Weaviate. The system allows uploading documents (PDF, DOCX, JSON, TXT), storing their embeddings in Weaviate, and querying the stored knowledge.

## Features

- **Multiple Document Types**: Support for PDF, DOCX, JSON, and TXT files
- **Weaviate Integration**: Uses Weaviate Cloud for vector storage
- **Document Management**: Replaces previous embeddings when the same file is uploaded again
- **File-Specific Querying**: Ability to query specific documents or all documents
- **Metadata-Rich Responses**: Answers include filename, source document snippets, and other identifiers
- **FastAPI Backend**: RESTful API for all operations

## Setup

### Prerequisites

- Python 3.12
- OpenAI API key(https://openai.com/)
- Weaviate Cloud account and API key(https://weaviate.io/)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/vinayjain18/rag-system.git
   cd rag-system
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file based on the provided `.env.sample`:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   WEAVIATE_URL=your_weaviate_cluster_url_here
   WEAVIATE_API_KEY=your_weaviate_api_key_here
   ```

### Running the Application

Start the FastAPI server:
```
uvicorn app:app --reload
```

The API will be available at `http://localhost:8005`.

## API Endpoints

### Upload Document
- **URL**: `/upload`
- **Method**: `POST`
- **Request**: Form data with file
- **Description**: Upload and process a document, replacing previous versions if the same filename is uploaded again.

### List Documents
- **URL**: `/documents`
- **Method**: `GET`
- **Description**: List all stored documents with their IDs.

### Query Documents
- **URL**: `/query`
- **Method**: `POST`
- **Request Body**:
  ```json
  {
    "question": "Your question here",
    "doc_ids": ["optional_doc_id_1", "optional_doc_id_2"]
  }
  ```
- **Description**: Query the system with a question, optionally filtering by specific document IDs.

### Delete Document
- **URL**: `/documents/{doc_id}`
- **Method**: `DELETE`
- **Description**: Delete a document and its chunks by document ID.

## Usage Examples

### Uploading a Document

```bash
curl --location 'http://localhost:8005/upload' --form 'file=@"/C:/Users/Vinay/Downloads/Untitled document.docx"'
```

### Querying Documents

```bash
curl --location 'http://localhost:8005/query' \
--header 'Content-Type: application/json' \
--data '{"question": "What are his hobbies?", 
"doc_ids": ["15d61ea2-132e-5662-86c6-3dac4762de88"]}'
```

## Architecture

The system follows this workflow:

1. **Document Upload**: Files are uploaded, processed based on their type, and chunked.
2. **Embedding Generation**: OpenAI embeddings are generated for each chunk.
3. **Vector Storage**: Embeddings and metadata are stored in Weaviate.
4. **Retrieval**: Relevant chunks are retrieved based on semantic similarity.
5. **Generation**: LLM generates answers based on retrieved chunks.
6. **Pipeline**: User can view the documents uploaded on the Weaviate.
