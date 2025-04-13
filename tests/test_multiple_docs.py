import pytest
import os
import shutil
from fastapi.testclient import TestClient
from app import app, DATA_DIR
from pathlib import Path

client = TestClient(app)

# Test data
doc1_content = """Passport Requirements:
1. Completed Form DS-11
2. Proof of U.S. Citizenship (e.g., birth certificate, naturalization certificate)
3. Valid Government-issued photo ID
4. Passport Photo (2x2 inches)
5. Payment for passport fees ($130 for passport book, $35 execution fee)"""

doc2_content = """Visa Requirements:
1. Completed Form DS-160
2. Passport valid for at least 6 months
3. One recent photograph (2x2 inches)
4. Receipt for visa application fee ($185)
5. Additional documents may be required based on visa type"""

class TestMultipleDocsQuery:
    @pytest.fixture(autouse=True)
    def setup(self):
        # Create test data directory
        self.test_data_dir = os.path.join(DATA_DIR, "test_user", "test_project")
        os.makedirs(self.test_data_dir, exist_ok=True)
        
        # Create test files
        self.doc1_path = os.path.join(self.test_data_dir, "passport_requirements.txt")
        self.doc2_path = os.path.join(self.test_data_dir, "visa_requirements.txt")
        
        # Write content to files
        with open(self.doc1_path, "w") as f:
            f.write(doc1_content)
        with open(self.doc2_path, "w") as f:
            f.write(doc2_content)
        
        # Upload files
        with open(self.doc1_path, "rb") as f:
            response = client.post("/upload", 
                                 files={"file": ("passport_requirements.txt", f, "text/plain")},
                                 params={"user": "test_user", "project": "test_project"})
            assert response.status_code == 200
        
        with open(self.doc2_path, "rb") as f:
            response = client.post("/upload", 
                                 files={"file": ("visa_requirements.txt", f, "text/plain")},
                                 params={"user": "test_user", "project": "test_project"})
            assert response.status_code == 200
        
        yield
        
        # Cleanup
        if os.path.exists(self.test_data_dir):
            shutil.rmtree(self.test_data_dir)

    def test_passport_question(self):
        """Test a question that should be answered from the passport document"""
        response = client.post("/query", 
                             json={"text": "What are the requirements for a passport application?"},
                             params={"user": "test_user", "project": "test_project"})
        assert response.status_code == 200
        data = response.json()
        
        # Check that the answer contains key passport requirements
        answer = data["answer"].lower()
        assert "form ds-11" in answer
        assert "citizenship" in answer
        assert "photo" in answer
        assert "fee" in answer
        
        # Verify sources include passport document
        sources = [s["file"] for s in data["sources"]]
        assert "passport_requirements.txt" in sources

    def test_visa_question(self):
        """Test a question that should be answered from the visa document"""
        response = client.post("/query", 
                             json={"text": "What are the requirements for a visa application?"},
                             params={"user": "test_user", "project": "test_project"})
        assert response.status_code == 200
        data = response.json()
        
        # Check that the answer contains key visa requirements
        answer = data["answer"].lower()
        assert "form ds-160" in answer
        assert "passport" in answer
        assert "photograph" in answer
        assert "fee" in answer
        
        # Verify sources include visa document
        sources = [s["file"] for s in data["sources"]]
        assert "visa_requirements.txt" in sources

    def test_cross_document_question(self):
        """Test a question that requires information from both documents"""
        response = client.post("/query", 
                             json={"text": "What documents and fees are needed for both passport and visa applications?"},
                             params={"user": "test_user", "project": "test_project"})
        assert response.status_code == 200
        data = response.json()
        
        # Check that the answer contains information from both documents
        answer = data["answer"].lower()
        
        # Passport requirements
        assert "form ds-11" in answer
        assert "passport fee" in answer or "$130" in answer
        
        # Visa requirements
        assert "form ds-160" in answer
        assert "visa fee" in answer or "$185" in answer
        
        # Verify both documents are included in sources
        sources = [s["file"] for s in data["sources"]]
        assert "passport_requirements.txt" in sources
        assert "visa_requirements.txt" in sources 