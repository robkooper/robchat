# RobChat

A powerful Retrieval-Augmented Generation (RAG) system for document question answering. RobChat allows you to upload documents and ask questions about their content, receiving AI-generated answers with source citations.

## Features

- **Document Support**: Upload and process multiple document types:
  - PDF files
  - Word documents (DOCX)
  - Text files (TXT)
  - HTML files
  - PowerPoint presentations (PPTX)
  - Excel spreadsheets (XLSX)

- **Advanced RAG Pipeline**:
  - Document chunking and indexing
  - Semantic search using embeddings
  - Context-aware answer generation
  - Source citations in responses

- **Modern Tech Stack**:
  - LangChain for RAG pipeline
  - Llama2 (7B-Chat) for text generation
  - sentence-transformers for embeddings
  - ChromaDB for vector storage
  - FastAPI for web API
  - React-based frontend

## Requirements

- Python 3.9+
- PyTorch
- 16GB RAM minimum (32GB recommended)
- GPU recommended but not required

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/robchat.git
   cd robchat
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the server:
   ```bash
   uvicorn app:app --reload
   ```

## Configuration

The application is configured using YAML files:

1. `config.yaml` - Main configuration file
2. `users.yaml` - User authentication configuration

### Configuration Files

#### config.yaml
The main configuration file contains settings for:
- Data storage
- Server configuration
- Model settings
- Vector store settings
- Authentication settings
- RAG settings

Example configuration:
```yaml
# Data Storage
data:
  base_directory: "data"
  supported_extensions:
    - ".pdf"
    - ".docx"
    - ".txt"
    - ".html"
    - ".pptx"
    - ".xlsx"

# Server Configuration
server:
  host: "0.0.0.0"
  port: 8000
  log_level: "info"

# Authentication Configuration
auth:
  secret_key: "your-secret-key-here"  # Change this in production!
  algorithm: "HS256"
  access_token_expire_minutes: 30

# Models Configuration
models:
  llm:
    repo_id: "TheBloke/Llama-2-7B-Chat-GGUF"
    filename: "llama-2-7b-chat.Q4_K_M.gguf"
    temperature: 0.7
    max_tokens: 2000
    context_window: 4096
    top_p: 0.95
    n_batch: 512
  embeddings:
    model_name: "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs:
      device: "cpu"

# Vector Store Configuration
vector_store:
  chunk_size: 1000
  chunk_overlap: 200
  retrieval_k: 4

# RAG Configuration
rag:
  qa_template: |
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer based on the context, just say "I don't know" - do not try to make up an answer.
    If you do know the answer, provide it clearly and concisely, incorporating relevant information from all provided sources.
    Make sure to use all relevant information from the sources to give a complete answer.

    Context:
    {context}

    Question: {question}

    Answer: Let me help you with that.
```

#### users.yaml
The users configuration file contains user authentication information. This file should be kept secure and not committed to version control. A template is provided in `users.example.yaml`.

Example configuration:
```yaml
admin:
  email: admin@example.com
  fullname: Administrator
  admin: true
  password: $2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW  # password: secret
test:
  email: test@example.com
  fullname: Test User
  admin: false
  password: $2b$12$UCqQEFTUM/fTwhlnSa9c8OLbUrrmWv1tfyR/mHh7eWPdAxvDP/JKK  # password: test
```

To set up users:
1. Copy `users.example.yaml` to `users.yaml`
2. Update the user information as needed
3. Generate new password hashes using the `get_password_hash` function in `app.py`

Note: The `users.yaml` file is automatically added to `.gitignore` to prevent accidental commits of sensitive information.

### Data Storage
- `base_directory`: Directory where uploaded files will be stored
- `supported_extensions`: List of file types that can be processed

### Server Configuration
- `host`: Server host address (default: "0.0.0.0")
- `port`: Server port (default: 8000)
- `log_level`: Logging verbosity (debug, info, warning, error, critical)

### Models Configuration
- `llm`: Large Language Model settings
  - `repo_id`: HuggingFace repository ID
  - `filename`: Model filename
  - `temperature`: Controls response randomness (0.0 to 1.0)
  - `max_tokens`: Maximum response length
  - `context_window`: Context size for processing
  - `top_p`: Nucleus sampling parameter
  - `n_batch`: Processing batch size
- `embeddings`: Embedding model settings
  - `model_name`: Model identifier
  - `device`: Processing device (cpu, cuda, mps)

### Vector Store Configuration
- `chunk_size`: Size of text chunks for processing
- `chunk_overlap`: Overlap between chunks
- `retrieval_k`: Number of text chunks to retrieve per query (default: 4). Note that this is not the same as the number of source documents, as chunks may come from the same document and are deduplicated in the response.

Note: The default values for these settings are optimized for typical use cases, but can be adjusted based on your needs. Increasing values like `context_window`, `retrieval_k`, or `chunk_size` may impact performance and memory usage. Adjust these settings based on your system's capabilities.

### Document Processing and Retrieval

The system handles large documents and document sets through a chunking and retrieval process:

1. **Document Chunking**:
   - Documents are split into smaller chunks (controlled by `chunk_size`)
   - Chunks overlap slightly (controlled by `chunk_overlap`)
   - This maintains context between chunks while making text manageable

2. **Retrieval Process**:
   - Only the top `retrieval_k` chunks are retrieved for each query
   - Chunks are selected based on semantic similarity to the query
   - Not all documents/chunks are searched at once
   - Focus is on the most relevant information

3. **Context Window**:
   - Limited by `context_window` setting (default: 4096 tokens)
   - Includes both query and retrieved chunks
   - May truncate content if limit is exceeded

This approach has both advantages and limitations:

**Advantages**:
- Efficient processing
- Faster response times
- Lower memory usage
- Focus on most relevant information

**Limitations**:
- May miss relevant information in non-top chunks
- Limited context for complex queries
- May not capture relationships between distant document parts

For large document sets, consider:
- Pre-filtering documents using metadata
- Using document summaries
- Implementing hierarchical search
- Using document clustering

Adjust settings in `config.yaml` to optimize for your needs:
```yaml
vector_store:
  chunk_size: 2000     # Increase chunk size
  chunk_overlap: 400   # Increase overlap
  retrieval_k: 8       # Retrieve more chunks

models:
  llm:
    context_window: 8192  # Increase context window
```

Note: Increasing these values will impact performance, memory usage, and response times.

## Examples

The `example` folder contains demonstration scripts and sample documents to help you get started with the API:

### Sample Documents
- `quantum_computing.txt`: Information about quantum computing concepts and applications
- `artificial_intelligence.txt`: Overview of AI technologies and their applications

### API Usage Examples

#### Python Example
The `test_api.py` script demonstrates how to:
- Authenticate with the API
- Upload documents
- Query the documents
- Handle responses

Run the Python example:
```bash
cd example
python test_api.py
```

#### Shell Script Example
The `test_api.sh` script provides the same functionality using curl commands:

Run the shell script:
```bash
cd example
./test_api.sh
```

Note: The shell script requires `curl` and `jq` to be installed.

### Example Queries
The example scripts demonstrate three types of queries:
1. Specific topic queries (e.g., "What are the key concepts in quantum computing?")
2. Application queries (e.g., "What are the main applications of AI?")
3. Combined topic queries (e.g., "How can AI help with quantum computing?")

## Usage

1. Start the server:
     ```bash
     uvicorn app:app --reload
     ```

2. Open your browser to `http://localhost:8000`

3. Upload documents using the web interface

4. Ask questions about your documents

## API Endpoints

All endpoints require authentication using a Bearer token. To get a token:

- `POST /token`
  - Authenticate and get access token
  - Body (form data): `username` and `password`
  - Returns: `{"access_token": "token", "token_type": "bearer"}`

All other endpoints follow the RESTful pattern `/api/{user}/{project}` where:
- `{user}`: The username
- `{project}`: The project name
- All requests must include the header: `Authorization: Bearer <token>`

Available endpoints:

- `GET /api/{user}/projects`
  - Lists all projects for a user and the current active project
  - Returns: `{"projects": ["project1", "project2"], "current_project": "project1"}`
  - Error responses:
    - 401: Not authenticated
    - 403: Cannot access another user's projects

- `GET /api/{user}/{project}/files`
  - Lists all files in the project
  - Returns: `{"files": ["file1.pdf", "file2.txt"]}`
  - Error responses:
    - 401: Not authenticated
    - 403: Cannot access another user's files

- `POST /api/{user}/{project}/files`
  - Upload a new file to the project
  - Accepts: multipart/form-data with file
  - Supported file types: .pdf, .docx, .txt, .html, .pptx, .xlsx
  - Returns: `{"filename": "file.txt", "chunks": 5, "initial_count": 0, "final_count": 5, "replaced_existing": false}`
  - Error responses:
    - 400: Unsupported file type
    - 401: Not authenticated
    - 403: Cannot upload to another user's project
    - 500: File processing error

- `DELETE /api/{user}/{project}/files/{filename}`
  - Delete a file from the project
  - Returns: `{"status": "success", "message": "File <filename> deleted successfully"}`
  - Error responses:
    - 401: Not authenticated
    - 403: Cannot delete another user's files
    - 404: File not found
    - 500: Deletion error

- `POST /api/{user}/{project}/query`
  - Query the project's documents
  - Body: `{"text": "your question here"}`
  - Returns: 
    ```json
    {
      "answer": "AI-generated answer",
      "sources": [
        {
          "number": 1,
          "file": "source1.pdf",
          "text": "Full source text",
          "preview": "First 150 characters..."
        }
      ],
      "metrics": {
        "time_seconds": 1.23,
        "total_tokens": 100,
        "input_tokens": 20,
        "output_tokens": 80
      }
    }
    ```
  - Error responses:
    - 401: Not authenticated
    - 403: Cannot query another user's documents
    - 404: No documents found
    - 422: Invalid query format
    - 500: Query processing error

## Project Structure

```
robchat/
├── app.py           # Main FastAPI application
├── static/          # Static files
│   ├── login.html   # Main entry point and login page
│   ├── chat.html    # Chat interface
│   ├── styles.css   # Global styles
│   ├── app.js       # Main application logic
│   ├── login.js     # Login functionality
│   └── favicon.svg  # Site icon
├── example/         # Example scripts and documents
│   ├── test_api.py  # Python API example
│   ├── test_api.sh  # Shell API example
│   ├── quantum_computing.txt    # Sample document
│   └── artificial_intelligence.txt  # Sample document
├── data/           # Document storage
│   └── {user}/
│       └── {project}/
│           └── chroma_db/  # Vector store
├── tests/          # Test files
├── config.yaml     # Application configuration
├── config.example.yaml  # Example configuration
├── requirements.txt # Python dependencies
└── CHANGELOG.md    # Version history
```

## Error Handling

- Automatic retry with exponential backoff for model operations
- Graceful handling of file processing errors
- Detailed error logging
- Clean file cleanup on processing failures

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details