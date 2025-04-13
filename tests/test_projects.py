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

def test_projects_endpoint_authenticated():
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
    response = client.get("/api/test/projects")
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"

def test_projects_endpoint_wrong_user():
    token = get_test_token()
    response = client.get(
        "/api/wrong_user/projects",
        headers={"Authorization": f"Bearer {token}"}
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Permission denied: Cannot access projects of another user"

def test_projects_endpoint_invalid_token():
    response = client.get(
        "/api/test/projects",
        headers={"Authorization": "Bearer invalid_token"}
    )
    assert response.status_code == 401
    assert response.json()["detail"] == "Could not validate credentials" 