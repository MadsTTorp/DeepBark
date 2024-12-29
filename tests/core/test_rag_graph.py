import pytest
from app.core.rag_graph import retrieve, State, AnswerWithSources
from langchain_core.documents import Document


def test_retrieve():
    # create a mock state with a sample question
    state = State(
        question="Hvilke farver har en Border Collie typisk?",
        context=[],
        answer=AnswerWithSources(answer="", sources=[]),
    )

    # call the retrieve function
    result = retrieve(state)

    # check that the result contains the expected keys
    assert "context" in result
    assert len(result["context"]) > 0
    assert isinstance(result["context"][0], Document)

    # check that the retrieved documents are relevant
    for doc in result["context"]:
        assert "sort" in doc.page_content.lower()


def test_retrieve_no_results():
    # Create a mock state with a question that has no relevant documents
    state = State(
        question="Hvor lang tid tager det at galloper til jupiter?",
        context=[],
        answer=AnswerWithSources(answer="", sources=[]),
    )

    # Call the retrieve function
    result = retrieve(state)

    # Check that the result contains the expected keys
    assert "context" in result
    assert len(result["context"]) == 0
    assert (
        result["answer"]["answer"]
        == "Jeg kender desværre ikke svaret på dit spørgsmål, \på baggrund af de artikler jeg har adgang til."
    )
    assert len(result["answer"]["sources"]) == 0
