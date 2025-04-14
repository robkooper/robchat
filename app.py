"""
RobChat: A RAG-based Document Question Answering System

This application implements a Retrieval-Augmented Generation (RAG) system that allows users to:
1. Upload various document types (PDF, DOCX, TXT, HTML, PPTX, XLSX)
2. Process and index document content
3. Query the documents using natural language
4. Get AI-generated answers with source citations

The system uses:
- LangChain for the RAG pipeline
- Llama2 (7B-Chat) for text generation
- sentence-transformers for embeddings
- ChromaDB for vector storage
- FastAPI for the web API
"""

from typing import List, Optional, Dict, Any, Annotated
import os
import logging
import time
import signal
import sys
from functools import wraps
from datetime import datetime, timedelta
import traceback

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt
import bcrypt
from passlib.context import CryptContext
import yaml
import pypdf
import torch
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from docx import Document
from bs4 import BeautifulSoup
from pptx import Presentation
from openpyxl import load_workbook
import chromadb

# Load configuration first, before setting up logging
def load_config(config_path: str = "config.yaml") -> Dict[Any, Any]:
    """
    Load configuration from YAML file and check for required files.

    Args:
        config_path (str): Path to configuration file

    Returns:
        dict: Configuration dictionary

    Exits:
        If config file or users.yaml doesn't exist
    """
    # Check for users.yaml first
    if not os.path.exists("users.yaml"):
        logger.error("users.yaml file not found. Please create a users.yaml file based on users.example.yaml.")
        sys.exit(1)

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        logger.error("Configuration file %s not found. Please create a config.yaml file.", config_path)
        sys.exit(1)

# Load configuration before setting up logging
config = load_config()

# Setup logging with config
logging.basicConfig(
    level=getattr(logging, config.get("server", {}).get("log_level", "INFO").upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.info("Loaded configuration from config.yaml")

# Set specific loggers to INFO level
logging.getLogger('python_multipart.multipart').setLevel(logging.INFO)
logging.getLogger('urllib3.connectionpool').setLevel(logging.INFO)

# Device configuration
"""Configure the compute device (CPU, CUDA, or MPS) based on availability"""
try:
    if torch.backends.mps.is_available():
        DEVICE = "mps"  # Apple Silicon (M1/M2) GPU
        logger.info("Using Apple Metal (MPS) device")
    elif torch.cuda.is_available():
        DEVICE = "cuda"
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
    else:
        DEVICE = "cpu"
        logger.info("Using CPU device")

    logger.info("PyTorch version: %s", torch.__version__)
    logger.info("Device: %s", DEVICE)
except Exception as e:
    logger.warning("Error detecting device, falling back to CPU: %s", str(e))
    DEVICE = "cpu"

# Print HuggingFace cache location
try:
    from huggingface_hub import scan_cache_dir
    from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE
    cache_info = scan_cache_dir()
    logger.info("HuggingFace cache directory: %s", HUGGINGFACE_HUB_CACHE)
    logger.info("Number of cached models: %d", len(cache_info.repos))
    logger.info("Cached repositories:")
    for repo in cache_info.repos:
        logger.info("- %s", repo.repo_id)
except Exception as e:
    logger.warning("Error scanning cache directory: %s", str(e))

# Authentication models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    email: str
    fullname: str
    admin: bool
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

# Authentication settings
SECRET_KEY = config["auth"]["secret_key"]
ALGORITHM = config["auth"]["algorithm"]
ACCESS_TOKEN_EXPIRE_MINUTES = config["auth"]["access_token_expire_minutes"]

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_password_hash(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(plain_password, hashed_password):
    if isinstance(hashed_password, str):
        hashed_password = hashed_password.encode('utf-8')
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password)

def load_users():
    """Load users from users.yaml file"""
    try:
        with open('users.yaml', 'r') as f:
            users = yaml.safe_load(f)
            return users
    except FileNotFoundError:
        logger.error("users.yaml file not found")
        return {}
    except Exception as e:
        logger.error("Error loading users.yaml: %s", str(e))
        return {}

def get_user(username: str):
    """Get user from users.yaml file"""
    users = load_users()
    if username in users:
        user_dict = users[username]
        user_dict["username"] = username
        user_dict["hashed_password"] = user_dict.pop("password")  # Map password to hashed_password
        return UserInDB(**user_dict)
    return None

def authenticate_user(username: str, password: str):
    """Authenticate user against users.yaml file"""
    user = get_user(username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError as exc:
        raise credentials_exception from exc
    user = get_user(username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Update global constants from config
DATA_DIR = config["data"]["base_directory"]

# Define the prompts
HUMAN_TEMPLATE = "Question: {question}"

# Initialize the RAG components
def retry_with_fallback(max_attempts=3):
    """
    Decorator that implements retry logic with exponential backoff.

    Args:
        max_attempts (int): Maximum number of retry attempts

    Returns:
        Function wrapper that implements the retry logic
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1}/{max_attempts} failed: {str(e)}")
                    if attempt == max_attempts - 1:  # Last attempt
                        logger.error(f"All attempts failed for {func.__name__}: {str(e)}")
                        raise
                    time.sleep(2 ** attempt)  # Exponential backoff
            return None
        return wrapper
    return decorator

app = FastAPI(title="RobChat API")

# Setup static paths
static_path = os.path.join(os.path.dirname(__file__), "static")

# Mount the static files
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Models
class Query(BaseModel):
    text: str

class FileList(BaseModel):
    files: List[str]

class UserProjects(BaseModel):
    """Model for user projects response."""
    projects: List[str]  # List of all projects
    current_project: str

# API Endpoints
@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()] = None,
    username: str = Form(None),
    password: str = Form(None)
):
    """Handle both OAuth2 form and regular form data"""
    if form_data is None and (username is None or password is None):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either OAuth2 form data or username/password must be provided"
        )

    if form_data is not None:
        username = form_data.username
        password = form_data.password

    user = authenticate_user(username, password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/")
async def read_root():
    """Serve the main application page"""
    return FileResponse("static/login.html")

@app.get("/chat")
async def read_chat():
    """Serve the chat interface page"""
    return FileResponse("static/chat.html")

def parse_pdf(file_path: str) -> str:
    """Parse PDF file and extract text content"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        logger.error("Error parsing PDF: %s", str(e))
        raise ValueError("Failed to parse PDF") from e

def process_file_for_vectorstore(file_path: str) -> List[str]:
    """Process file and prepare it for vector store"""
    try:
        file_extension = os.path.splitext(file_path)[1].lower()

        if file_extension == '.pdf':
            text = parse_pdf(file_path)
        elif file_extension == '.docx':
            doc = Document(file_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        elif file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        elif file_extension == '.html':
            with open(file_path, 'r', encoding='utf-8') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')
                text = soup.get_text()
        elif file_extension == '.pptx':
            prs = Presentation(file_path)
            text = ""
            for slide in prs.slides:
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text += shape.text + "\n"
        elif file_extension == '.xlsx':
            wb = load_workbook(file_path)
            text = ""
            for sheet in wb:
                for row in sheet.iter_rows():
                    text += " ".join([str(cell.value) for cell in row if cell.value]) + "\n"
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)
        return chunks

    except Exception as e:
        logger.error("Error processing file: %s", str(e))
        raise ValueError(f"Failed to process file: {str(e)}") from e

@app.get("/api/{user}/{project}/files")
async def list_files(
    user: str,
    project: str,
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    """List files in a project"""
    if user != current_user.username:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied: Cannot access files of another user"
        )

    try:
        project_path = os.path.join(DATA_DIR, user, project)
        if not os.path.exists(project_path):
            return {"files": []}

        files = []
        for filename in os.listdir(project_path):
            if filename.endswith(('.txt', '.pdf', '.docx', '.html', '.pptx', '.xlsx')):
                file_path = os.path.join(project_path, filename)
                files.append({
                    "filename": filename,
                    "size": os.path.getsize(file_path),
                    "modified": datetime.fromtimestamp(os.path.getmtime(file_path)).isoformat()
                })

        return {"files": files}

    except Exception as e:
        logger.error("Error listing files: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list files: {str(e)}"
        ) from e

@app.post("/api/{user}/{project}/files")
async def create_file(
    user: str,
    project: str,
    current_user: Annotated[User, Depends(get_current_active_user)],
    file: UploadFile = File(...)
):
    """Upload a file to a project"""
    if user != current_user.username:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied: Cannot upload files for another user"
        )

    try:
        # Validate file extension
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in ['.txt', '.pdf', '.docx', '.html', '.pptx', '.xlsx']:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file type. Supported types are: .pdf, .docx, .txt, .html, .pptx, .xlsx"
            )

        # Create project directory if it doesn't exist
        project_path = os.path.join(DATA_DIR, user, project)
        os.makedirs(project_path, exist_ok=True)

        # Save the file
        file_path = os.path.join(project_path, file.filename)
        with open(file_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        # Process the file for vector store
        try:
            chunks = process_file_for_vectorstore(file_path)
            logger.info("Processed %d chunks from file %s", len(chunks), file.filename)
        except Exception as e:
            logger.error("Error processing file for vector store: %s", str(e))
            # Don't fail the upload if processing fails, just log the error

        return {
            "filename": file.filename,
            "size": os.path.getsize(file_path)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error uploading file: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}"
        ) from e

@app.post("/api/{user}/{project}/query")
async def query_rag(
    query: Query,
    user: str,
    project: str,
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    """Query the RAG system"""
    if user != current_user.username:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied: Cannot query documents of another user"
        )

    try:
        if not query.text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Query cannot be empty"
            )

        # Get vector store
        vector_store = get_vector_store(user, project)

        # Create retriever
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=get_llm(),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )

        # Run query
        result = qa_chain({"query": query.text})

        # Format response
        response = {
            "response": result["result"],
            "sources": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
                for doc in result["source_documents"]
            ]
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing query: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while processing your query: {str(e)}"
        ) from e

@app.get("/api/{user}/projects")
async def get_user_projects(
    user: str,
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    """Get list of projects for a user"""
    if user != current_user.username:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied: Cannot access projects of another user"
        )

    try:
        user_path = os.path.join(DATA_DIR, user)
        if not os.path.exists(user_path):
            return {"projects": [], "current_project": ""}

        projects = [d for d in os.listdir(user_path) if os.path.isdir(os.path.join(user_path, d))]
        current_project = projects[0] if projects else ""

        return {
            "projects": projects,
            "current_project": current_project
        }

    except Exception as e:
        logger.error("Error getting user projects: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user projects: {str(e)}"
        ) from e

@app.delete("/api/{user}/{project}/files/{filename}")
async def delete_file(
    user: str,
    project: str,
    filename: str,
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    """Delete a file from a project"""
    if user != current_user.username:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permission denied: Cannot delete files of another user"
        )

    try:
        file_path = os.path.join(DATA_DIR, user, project, filename)
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="File not found"
            )

        os.remove(file_path)
        return {"message": f"File {filename} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting file: %s", str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        ) from e

@app.get("/api/me", response_model=User)
async def get_current_user_details(
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    """Get current user details"""
    return current_user

def handle_exit(signum, frame):
    """Handle graceful shutdown"""
    logger.info("Received shutdown signal")
    sys.exit(0)

signal.signal(signal.SIGINT, handle_exit)
signal.signal(signal.SIGTERM, handle_exit)

if __name__ == "__main__":
    import uvicorn
    import signal
    import sys

    # Configure uvicorn logging
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.setLevel(logging.INFO)

    # Global flag to control server state
    should_exit = False

    def handle_exit(signum, frame):
        """Handle exit signals"""
        global should_exit
        should_exit = True
        logger.info("Shutdown requested. Stopping server...")
        sys.exit(0)

    # Register signal handlers
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)

    # Use server settings from config
    config = uvicorn.Config(
        app=app,
        host=config["server"]["host"],
        port=config["server"]["port"],
        log_level=config["server"]["log_level"]
    )
    server = uvicorn.Server(config)
    server.run()
