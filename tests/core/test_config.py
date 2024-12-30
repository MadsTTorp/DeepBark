import os
import pytest
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from app.core.config import llm, custom_rag_prompt


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


def test_prompt_template_initialization():
    assert isinstance(custom_rag_prompt, PromptTemplate)
    expected_template = (
    "Brug følgende stykker kontekst til at besvare spørgsmålet i slutningen. "
    "Hvis du ikke kender svaret, så sig bare, at du ikke ved det, og prøv "
    "ikke at opdigte et svar. Svar med maksimalt tre sætninger og hold svaret "
    "så kortfattet men præcist som muligt. Vær høflig i dit svar.\n\n"
    "{context}\n\nSpørgsmål: {question}\n\nHjælpsomt svar:"
    )  # Updated expected template string
    assert custom_rag_prompt.template == expected_template
