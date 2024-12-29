import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_ask_question():
    response = client.post("/rag/ask", json={"question": "Hvor mange ben har en hund?"})
    assert response.status_code == 200
    assert "answer" in response.json()
    # assert "sources" in response.json()["answer"]
