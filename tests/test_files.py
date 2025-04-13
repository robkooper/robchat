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

def test_files_endpoint_wrong_user():
    token = get_test_token()
    response = client.get(
        "/api/wrong_user/test_project/files",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Permission denied: Cannot access files of another user"

def test_upload_file_wrong_user():
    token = get_test_token()
    test_content = "This is a test file content"
    response = client.post(
        "/api/wrong_user/test_project/files",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("test.txt", test_content, "text/plain")}
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Permission denied: Cannot upload files for another user"

def test_query_wrong_user():
    token = get_test_token()
    response = client.post(
        "/api/wrong_user/test_project/query",
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "test query"}
    )
    assert response.status_code == 422
    assert "detail" in response.json()

def test_delete_file_wrong_user():
    token = get_test_token()
    response = client.delete(
        "/api/wrong_user/test_project/files/test.txt",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Permission denied: Cannot delete files of another user"

def test_upload_and_delete_file():
    token = get_test_token()
    test_content = "This is a test file content"
    
    # Upload file
    response = client.post(
        "/api/test/test_project/files",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("test.txt", test_content, "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "filename" in data
    assert "chunks" in data
    assert data["filename"] == "test.txt"
    assert isinstance(data["chunks"], int)
    assert data["chunks"] > 0
    
    # Delete file
    response = client.delete(
        "/api/test/test_project/files/test.txt",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["message"] == "File test.txt deleted successfully"

def test_delete_nonexistent_file():
    token = get_test_token()
    response = client.delete(
        "/api/test/test_project/files/nonexistent.txt",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "File not found"

def test_upload_invalid_file():
    token = get_test_token()
    test_content = "This is a test file content"
    response = client.post(
        "/api/test/test_project/files",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("test.invalid", test_content, "application/octet-stream")}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported file type. Supported types are: .pdf, .docx, .txt, .html, .pptx, .xlsx" 