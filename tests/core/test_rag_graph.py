from app.core.rag_graph import retrieve, generate, query_or_respond, MessagesState
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage


def test_retrieve():
    # Create a mock state with a sample question
    state = MessagesState(
        messages=[
            HumanMessage(content="Er en border collie en god familie hund?")
        ]
    )

    # Call the retrieve function
    result = retrieve("Er en border collie en god familie hund?")

    # Check that the result contains the expected keys
    assert isinstance(result, tuple)
    assert len(result) == 2
    serialized, retrieved_docs = result
    assert isinstance(serialized, str)
    assert isinstance(retrieved_docs, list)
    assert len(retrieved_docs) > 0
    assert isinstance(retrieved_docs[0], Document)

    # Check that the retrieved documents are relevant
    for doc in retrieved_docs:
        assert "border collie" in doc.page_content.lower()
