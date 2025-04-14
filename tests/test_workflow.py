import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def get_test_token():
    response = client.post(
        "/token",
        data={"username": "test", "password": "test"}
    )
    return response.json()["access_token"]

@pytest.fixture
def test_file_content():
    return """
    The Zylax programming language was created by Dr. Elara Voss in 2025.
    It was first released in 2026 as an open-source project.
    Zylax is known for its unique quantum-inspired syntax and parallel processing capabilities.
    The language supports a novel programming paradigm called "quantum imperative".
    Zylax is primarily used in quantum computing simulations and advanced AI research.
    Its most notable feature is the ability to write code that runs both forwards and backwards in time.
    """

@pytest.fixture
def uploaded_file(test_file_content):
    token = get_test_token()
    response = client.post(
        "/api/test/test_project/files",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("zylax_info.txt", test_file_content, "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["filename"] == "zylax_info.txt"
    return token, data["filename"]

def test_file_upload(test_file_content):
    token = get_test_token()
    response = client.post(
        "/api/test/test_project/files",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("zylax_info.txt", test_file_content, "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "filename" in data
    assert "chunks" in data
    assert data["filename"] == "zylax_info.txt"
    assert isinstance(data["chunks"], int)
    assert data["chunks"] > 0

def test_query_with_known_answer(uploaded_file):
    token, filename = uploaded_file
    response = client.post(
        "/api/test/test_project/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"text": "Who created Zylax?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "metrics" in data
    assert "Elara Voss" in data["answer"]
    assert len(data["sources"]) > 0
    assert filename in data["sources"][0]["file"]

def test_query_with_unknown_answer(uploaded_file):
    token, _ = uploaded_file
    response = client.post(
        "/api/test/test_project/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"text": "What is the capital of France?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    # Check for any variation of "I don't know" in the answer
    assert any(phrase in data["answer"].lower() for phrase in ["i don't know", "i do not know", "based on the context"])

def test_file_deletion(uploaded_file):
    token, filename = uploaded_file
    response = client.delete(
        f"/api/test/test_project/files/{filename}",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["message"] == f"File {filename} deleted successfully"

def test_query_after_deletion(uploaded_file):
    token, filename = uploaded_file
    # Delete the file first
    response = client.delete(
        f"/api/test/test_project/files/{filename}",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    
    # Try to query after deletion
    response = client.post(
        "/api/test/test_project/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"text": "Who created Zylax?"}
    )
    # The application might return either a 404 or a 500 error when no documents are found
    assert response.status_code in [404, 500]
    if response.status_code == 404:
        assert response.json()["detail"] == "No documents found in the project"
    else:
        assert "error" in response.json()["detail"].lower() 