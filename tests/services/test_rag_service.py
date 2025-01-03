from unittest.mock import patch
from app.services.rag_service import get_rag_answer
from langchain_core.messages import AIMessage, ToolMessage
from app.core.config import memory_config


@patch("app.services.rag_service.graph")
def test_get_rag_answer(mock_graph):
    # Create a mock AIMessage and ToolMessage
    mock_ai_message = AIMessage(content="This is a sample answer.")
    mock_tool_message = ToolMessage(
        content=(
            "Source: {'source': 'http://example.com', 'start_index': 0, "
            "'section': None}\nContent: Sample content"
        ),
        artifact=[
            {
                "metadata": {"source": "http://example.com"},
                "page_content": "Sample content",
            }
        ]
    )

    # Define the mock response from the graph
    mock_response = [{"messages": [mock_tool_message, mock_ai_message]}]
    mock_graph.stream.return_value = mock_response

    # Call the get_rag_answer function
    question = "What is the best dog food?"
    result = get_rag_answer(question)

    # Verify that the graph.stream method was called with the correct params
    mock_graph.stream.assert_called_once_with(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
        config=memory_config,
    )

    # Verify the result
    assert result["answer"] == "This is a sample answer."
    assert result["sources"] == ["http://example.com"]


def test_get_rag_answer_no_sources():
    # Create a mock AIMessage without ToolMessage
    mock_ai_message = AIMessage(content="This is a sample answer.")

    # Define the mock response from the graph
    mock_response = [{"messages": [mock_ai_message]}]

    with patch("app.services.rag_service.graph.stream",
               return_value=mock_response):
        # Call the get_rag_answer function
        question = "What is the best dog food?"
        result = get_rag_answer(question)

        # Verify the result
        assert result["answer"] == ("Jeg kender desværre ikke svaret "
                                    "på dit spørgsmål, på baggrund af "
                                    "de artikler jeg har adgang til.")
        assert result["sources"] == []


def test_get_rag_answer_error_handling():

    # Create a mock ToolMessage
    mock_tool_message = ToolMessage(
        content="",
        name="retrieve",
        artifact=[]
        )
    # Create a mock AIMessage
    mock_ai_message = AIMessage(
        content="This is a sample answer.",
        additional_kwargs={'refusal': None},
        )

    # Define the mock response from the graph
    mock_response = [{"messages": [mock_tool_message, mock_ai_message]}]

    with patch("app.services.rag_service.graph.stream",
               return_value=mock_response):
        # Call the get_rag_answer function
        question = "What is the best dog food?"
        result = get_rag_answer(question)

        # Verify the result
        assert result["answer"] == "Noget gik galt, prøv venligst igen."
        assert result["sources"] == []
