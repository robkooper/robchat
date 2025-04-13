import os
import shutil
import logging
import bcrypt
from fastapi.testclient import TestClient
from app import app, DATA_DIR, fake_users_db, create_access_token
from datetime import timedelta

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class TestPassportQuery:
    def setup_method(self):
        """Set up test environment"""
        self.client = TestClient(app)
        self.test_user = "test_user"
        self.test_project = "test_project"
        
        # Add test user to fake_users_db
        fake_users_db[self.test_user] = {
            "username": self.test_user,
            "hashed_password": bcrypt.hashpw("test".encode('utf-8'), bcrypt.gensalt()),
            "disabled": False,
        }
        
        # Create test data directory
        self.test_dir = os.path.join(DATA_DIR, self.test_user, self.test_project)
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create a test document with passport information
        self.test_content = """
        Requirements for a U.S. Passport Application:

        1. Completed Form DS-11
        2. Proof of U.S. Citizenship (e.g., birth certificate, naturalization certificate)
        3. Valid Government-issued photo ID
        4. Passport Photo (2x2 inches)
        5. Payment for passport fees
           - $130 for passport book
           - $35 execution fee
        
        Additional Requirements:
        - Must appear in person at a passport acceptance facility
        - Photos must be recent (within 6 months)
        - Original documents or certified copies required
        - Payment must be made separately for application and execution fees
        """
        
        self.test_file = os.path.join(self.test_dir, "passport_info.txt")
        with open(self.test_file, "w") as f:
            f.write(self.test_content)
            
        # Get auth token for test user
        access_token = create_access_token(
            data={"sub": self.test_user},
            expires_delta=timedelta(minutes=30)
        )
        self.headers = {"Authorization": f"Bearer {access_token}"}

    def teardown_method(self):
        """Clean up test environment"""
        if os.path.exists(os.path.join(DATA_DIR, self.test_user)):
            shutil.rmtree(os.path.join(DATA_DIR, self.test_user))
        # Remove test user from fake_users_db
        if self.test_user in fake_users_db:
            del fake_users_db[self.test_user]

    def test_passport_query(self, capsys):
        """Test querying about passport requirements"""
        # First, ensure the file is processed into the vector store
        with open(self.test_file, "rb") as f:
            response = self.client.post(
                f"/api/{self.test_user}/{self.test_project}/files",
                files={"file": ("passport_info.txt", f, "text/plain")},
                headers=self.headers
            )
        assert response.status_code == 200
        
        # Now query about passport requirements
        query_response = self.client.post(
            f"/api/{self.test_user}/{self.test_project}/query",
            json={"text": "what is needed for a passport"},
            headers=self.headers
        )
        
        assert query_response.status_code == 200
        
        result = query_response.json()
        
        # Print the query result
        with capsys.disabled():
            print("\nQuery Result:")
            print("Answer:", result["answer"])
            print("\nSources:")
            for source in result["sources"]:
                print(f"- {source['file']}: {source['preview']}")
            print("\nMetrics:", result["metrics"])
        
        assert "answer" in result
        assert "sources" in result
        
        # Check that the answer contains key passport requirements
        answer = result["answer"].lower()
        expected_items = ["form ds-11", "citizenship", "photo", "fee"]
        for item in expected_items:
            assert item in answer, f"Answer should mention {item}"
        
        # Verify sources are provided
        assert len(result["sources"]) > 0
        assert result["sources"][0]["file"] == "passport_info.txt" 