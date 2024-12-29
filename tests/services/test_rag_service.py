from unittest.mock import patch, MagicMock
from app.services.rag_service import get_rag_answer


@patch("app.services.rag_service.get_graph")
def test_get_rag_answer(mock_get_graph):
    # Create a mock graph object
    mock_graph = MagicMock()
    mock_get_graph.return_value = mock_graph

    # Define the mock response from the graph
    mock_response = {
        "context": [{"page_content": "Sample content"}],
        "answer": {
            "answer": "This is a sample answer.",
            "sources": ["http://example.com"],
        },
    }
    mock_graph.invoke.return_value = mock_response

    # Call the get_rag_answer function
    question = "What is the best dog food?"
    result = get_rag_answer(question)

    # Verify that the graph.invoke method was called with the correct params
    mock_graph.invoke.assert_called_once_with({"question": question})

    # Verify the result
    assert result["context"] == mock_response["context"]
    assert result["answer"] == mock_response["answer"]
    assert result["answer"]["answer"] == "This is a sample answer."
    assert result["answer"]["sources"] == ["http://example.com"]
