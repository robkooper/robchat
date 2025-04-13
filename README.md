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

The application is configured using a `config.yaml` file. A template is provided in `config.example.yaml`. Copy this file to `config.yaml` and customize it for your environment:

```bash
cp config.example.yaml config.yaml
```

The configuration file contains the following sections:

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

All endpoints follow the RESTful pattern `/api/{user}/{project}` where:
- `{user}`: The username
- `{project}`: The project name

Available endpoints:

- `GET /api/{user}/projects`
  - Lists all projects for a user
  - Returns: `{"projects": ["project1", "project2"], "current_project": "project1"}`

- `POST /api/{user}/switch`
  - Switch to a different project
  - Body: `{"user": "username", "project": "project_name"}`
  - Returns: Success message and project directory info

- `GET /api/{user}/{project}/files`
  - Lists all files in the project
  - Returns: `{"files": ["file1.pdf", "file2.txt"]}`

- `POST /api/{user}/{project}/files`
  - Upload a new file to the project
  - Accepts: multipart/form-data with file
  - Returns: File processing info including chunk count

- `POST /api/{user}/{project}/query`
  - Query the project's documents
  - Body: `{"text": "your question here"}`
  - Returns: AI-generated answer with source citations

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