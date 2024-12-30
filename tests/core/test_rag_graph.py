from app.core.rag_graph import retrieve, generate, State, AnswerWithSources
from langchain_core.documents import Document


def test_retrieve():
    # create a mock state with a sample question
    state = State(
        question="Er en border collie en god familie hund?",
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
        assert "border collie" in doc.page_content.lower()


def test_rag_when_no_results_found():
    # create a mock state with a question that has no relevant documents
    state = State(
        question="Hvor lang tid tager det at galloper til jupiter?",
        context=[],
        answer=AnswerWithSources(answer="", sources=[]),
    )

    # call the generate function
    result = generate(state)

    # Check that the result contains the expected keys
    assert "answer" in result
    assert (
        result["answer"]["answer"] ==
        "Jeg kender desværre ikke svaret på dit spørgsmål, "
        "på baggrund af de artikler jeg har adgang til."
    )
    assert len(result["answer"]["sources"]) == 0
