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
- `retrieval_k`: Number of chunks to retrieve per query

## Usage

1. Start the server:
    ```bash
    python app.py
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
├── frontend/        # Web interface
│   ├── index.html
│   ├── styles.css
│   └── app.js
├── data/           # Document storage
│   └── {user}/
│       └── {project}/
│           └── chroma_db/  # Vector store
└── requirements.txt
```

## Error Handling

- Automatic retry with exponential backoff for model operations
- Graceful handling of file processing errors
- Detailed error logging
- Clean file cleanup on processing failures

## Limitations

- Context window limited to 4096 tokens
- Maximum 2 source documents per query for performance
- Some document types may have limited text extraction

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details