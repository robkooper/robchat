import requests
import json
import os
from pathlib import Path

# API Configuration
BASE_URL = "http://localhost:8000"
USERNAME = "test"
PASSWORD = "test"
PROJECT = "test_project"

def login():
    """Login and get access token"""
    response = requests.post(
        f"{BASE_URL}/token",
        data={"username": USERNAME, "password": PASSWORD}
    )
    if response.status_code != 200:
        raise Exception(f"Login failed: {response.text}")
    return response.json()["access_token"]

def upload_file(token, file_path):
    """Upload a file to the project"""
    headers = {"Authorization": f"Bearer {token}"}
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "text/plain")}
        response = requests.post(
            f"{BASE_URL}/api/{USERNAME}/{PROJECT}/files",
            headers=headers,
            files=files
        )
    if response.status_code != 200:
        raise Exception(f"File upload failed: {response.text}")
    return response.json()

def query(token, question):
    """Query the documents"""
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    response = requests.post(
        f"{BASE_URL}/api/{USERNAME}/{PROJECT}/query",
        headers=headers,
        json={"text": question}
    )
    if response.status_code != 200:
        raise Exception(f"Query failed: {response.text}")
    return response.json()

def main():
    try:
        # Login and get token
        print("Logging in...")
        token = login()
        print("Login successful!")

        # Upload files
        print("\nUploading files...")
        for file_name in ["quantum_computing.txt", "artificial_intelligence.txt"]:
            file_path = Path(__file__).parent / file_name
            result = upload_file(token, file_path)
            print(f"Uploaded {file_name}: {result}")

        # Make some queries
        print("\nMaking queries...")

        # Query 1: Quantum computing basics
        print("\nQuery 1: What are the key concepts in quantum computing?")
        result = query(token, "What are the key concepts in quantum computing?")
        print("Answer:", result["answer"])
        print("Sources:", [s["file"] for s in result["sources"]])

        # Query 2: AI applications
        print("\nQuery 2: What are the main applications of AI?")
        result = query(token, "What are the main applications of AI?")
        print("Answer:", result["answer"])
        print("Sources:", [s["file"] for s in result["sources"]])

        # Combined query
        print("\nQuery 3: How can AI help with quantum computing?")
        result = query(token, "How can AI help with quantum computing?")
        print("Answer:", result["answer"])
        print("Sources:", [s["file"] for s in result["sources"]])

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
