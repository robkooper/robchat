"""Test suite for the complete workflow of the application.

This module contains tests that verify the complete workflow of the application,
including file upload, querying, and deletion operations. It ensures that all
components work together correctly and maintain proper state throughout the process.
"""

import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def get_test_token():
    """Get a test authentication token.
    
    Returns:
        str: A JWT token for authentication.
    """
    response = client.post(
        "/token",
        data={"username": "test", "password": "test"}
    )
    return response.json()["access_token"]

@pytest.fixture
def test_file_content():
    """Fixture providing test file content.
    
    Returns:
        str: Content for test files.
    """
    return """Zylax Programming Language Documentation

Zylax is a revolutionary programming language created by Dr. Elara Voss in 2023.
It combines the best features of functional and imperative programming paradigms
while introducing novel concepts in quantum computing integration.

Key Features:
- Type-safe quantum computing primitives
- Automatic parallelization of computational tasks
- Built-in support for distributed systems
- Advanced pattern matching capabilities
- Zero-cost abstractions for high-performance computing

The language is designed to be both powerful and accessible, making it suitable
for both academic research and industrial applications."""

@pytest.fixture
def uploaded_file(test_file_content):
    """Fixture that uploads a test file and returns its filename.
    
    Args:
        test_file_content (str): Content to write to the test file.
        
    Returns:
        str: Name of the uploaded file.
    """
    token = get_test_token()
    response = client.post(
        "/api/test/test_project/files",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("zylax_info.txt", test_file_content, "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    return data["filename"]

def test_file_upload(test_file_content):
    """Test the file upload functionality.
    
    Args:
        test_file_content (str): Content to upload in the test file.
    """
    token = get_test_token()
    response = client.post(
        "/api/test/test_project/files",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("zylax_info.txt", test_file_content, "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "filename" in data
    assert data["filename"] == "zylax_info.txt"
    assert "chunks" in data
    assert data["chunks"] > 0

def test_query_document(uploaded_file):
    """Test querying an uploaded document.
    
    Args:
        uploaded_file (str): Name of the uploaded file to query.
    """
    token = get_test_token()
    response = client.post(
        "/api/test/test_project/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "Who created Zylax?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "Elara Voss" in data["answer"]
    assert uploaded_file in data["sources"][0]["file"]

def test_delete_file(uploaded_file):
    """Test deleting an uploaded file.
    
    Args:
        uploaded_file (str): Name of the file to delete.
    """
    token = get_test_token()
    response = client.delete(
        f"/api/test/test_project/files/{uploaded_file}",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == f"File {uploaded_file} deleted successfully"

def test_query_after_deletion(uploaded_file):
    """Test querying after file deletion should fail.
    
    Args:
        uploaded_file (str): Name of the file that was deleted.
    """
    token = get_test_token()
    response = client.post(
        "/api/test/test_project/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "What are the key features of Zylax?"}
    )
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "No documents found" in data["detail"]

def test_complete_workflow(test_file_content):
    """Test the complete workflow: upload, query, and delete.
    
    Args:
        test_file_content (str): Content to use for the test file.
    """
    # Upload file
    token = get_test_token()
    response = client.post(
        "/api/test/test_project/files",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("zylax_info.txt", test_file_content, "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    filename = data["filename"]
    assert filename == "zylax_info.txt"

    # Query file
    response = client.post(
        "/api/test/test_project/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "What is Zylax?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert filename in data["sources"][0]["file"]

    # Delete file
    response = client.delete(
        f"/api/test/test_project/files/{filename}",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert data["message"] == f"File {filename} deleted successfully"

    # Verify file is deleted
    response = client.post(
        "/api/test/test_project/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "What is Zylax?"}
    )
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "No documents found" in data["detail"]
