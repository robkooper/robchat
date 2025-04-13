import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_token_valid_credentials():
    """Test token endpoint with valid credentials"""
    response = client.post(
        "/token",
        data={"username": "test", "password": "test"}
    )
    assert response.status_code == 200
    assert "access_token" in response.json()
    assert response.json()["token_type"] == "bearer"

def test_token_invalid_username():
    """Test token endpoint with invalid username"""
    response = client.post(
        "/token",
        data={"username": "wrong", "password": "test"}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect username or password"

def test_token_invalid_password():
    """Test token endpoint with invalid password"""
    response = client.post(
        "/token",
        data={"username": "test", "password": "wrong"}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Incorrect username or password"

def test_token_missing_credentials():
    """Test token endpoint with missing credentials"""
    response = client.post("/token")
    assert response.status_code == 422  # FastAPI validation error

def get_test_token():
    """Helper function to get a valid token for test user"""
    response = client.post(
        "/token",
        data={"username": "test", "password": "test"}
    )
    return response.json()["access_token"]

def test_projects_endpoint_authenticated():
    """Test projects endpoint with valid authentication"""
    token = get_test_token()
    response = client.get(
        "/api/test/projects",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "current_project" in data
    assert "projects" in data
    assert isinstance(data["projects"], list)

def test_projects_endpoint_no_auth():
    """Test projects endpoint without authentication"""
    response = client.get("/api/test/projects")
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"

def test_projects_endpoint_wrong_user():
    """Test projects endpoint with valid token but wrong username"""
    token = get_test_token()
    response = client.get(
        "/api/wrong_user/projects",  # Try to access different user's projects
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Permission denied: Cannot access projects of another user"

def test_projects_endpoint_invalid_token():
    """Test projects endpoint with invalid token"""
    response = client.get(
        "/api/test/projects",
        headers={"Authorization": "Bearer invalid_token"}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Could not validate credentials"

def test_files_endpoint_wrong_user():
    """Test files endpoint with valid token but wrong username"""
    token = get_test_token()
    response = client.get(
        "/api/wrong_user/test_project/files",  # Try to access different user's files
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Permission denied: Cannot access files of another user"

def test_upload_file_wrong_user():
    """Test file upload endpoint with valid token but wrong username"""
    token = get_test_token()
    test_content = "This is a test file content"
    response = client.post(
        "/api/wrong_user/test_project/files",  # Try to upload to different user's project
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("test.txt", test_content, "text/plain")}
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Permission denied: Cannot upload files for another user"

def test_query_wrong_user():
    """Test query endpoint with valid token but wrong username"""
    token = get_test_token()
    response = client.post(
        "/api/wrong_user/test_project/query",  # Try to query different user's documents
        headers={"Authorization": f"Bearer {token}"},
        json={"query": "test query"}
    )
    assert response.status_code == 422
    assert "detail" in response.json()

def test_delete_file_wrong_user():
    """Test file deletion endpoint with valid token but wrong username"""
    token = get_test_token()
    response = client.delete(
        "/api/wrong_user/test_project/files/test.txt",  # Try to delete different user's file
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Permission denied: Cannot delete files of another user"

def test_upload_and_delete_file():
    """Test uploading a file and then deleting it"""
    token = get_test_token()
    
    # Upload a test file
    test_content = "This is a test file content"
    response = client.post(
        "/api/test/test_project/files",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("test.txt", test_content, "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "filename" in data
    assert data["filename"] == "test.txt"
    assert "chunks" in data
    assert data["chunks"] > 0
    
    # Delete the uploaded file
    response = client.delete(
        "/api/test/test_project/files/test.txt",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 200
    assert response.json()["message"] == "File test.txt deleted successfully"

def test_delete_nonexistent_file():
    """Test deleting a file that doesn't exist"""
    token = get_test_token()
    
    response = client.delete(
        "/api/test/test_project/files/nonexistent.txt",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 404
    assert response.json()["detail"] == "File not found"

def test_upload_invalid_file():
    """Test uploading an invalid file type"""
    token = get_test_token()
    
    test_content = "This is a test file content"
    response = client.post(
        "/api/test/test_project/files",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("test.invalid", test_content, "application/octet-stream")}
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported file type. Supported types are: .pdf, .docx, .txt, .html, .pptx, .xlsx" 