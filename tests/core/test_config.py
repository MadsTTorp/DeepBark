import os
import pytest
from langchain_openai import ChatOpenAI
from app.core.config import llm


def test_environment_variables_loaded():
    assert os.getenv("OPENAI_API_KEY") is not None


def test_llm_initialization():
    assert isinstance(llm, ChatOpenAI)
    assert llm.model_name == "gpt-4o-mini"


def test_llm_api_key():
    try:
        # Attempt to make a simple call to ensure the API key is valid
        response = llm.invoke("Hello, how are you?")
        assert response is not None
    except Exception as e:
        pytest.fail(f"Failed to init ChatOpenAI with the provided API key:{e}")
