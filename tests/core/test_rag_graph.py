import numpy as np
from unittest.mock import patch
from app.core.rag_graph import retrieve
from langchain_core.documents import Document


@patch("app.core.rag_graph.OpenAIEmbeddings.embed_text")
@patch("app.core.rag_graph.index.search")
def test_retrieve(mock_search, mock_embed_text):
    # Mock the embedding function
    mock_embed_text.return_value = np.array([0.1, 0.2, 0.3])

    # Mock the search function
    mock_search.return_value = (
        np.array([[0.1, 0.2, 0.3]]),  # distances
        np.array([[0, 1, 2]])         # indices
    )

    # Mock documents
    mock_documents = [
        Document(metadata={"source": "http://example.com/doc1"},
                 page_content="Content of document 1"),
        Document(metadata={"source": "http://example.com/doc2"},
                 page_content="Content of document 2"),
        Document(metadata={"source": "http://example.com/doc3"},
                 page_content="Content of document 3")
    ]

    # Patch the documents and similarity threshold
    with patch("app.core.rag_graph.documents", mock_documents), \
         patch("app.core.rag_graph.similarity_threshold", 0.15):

        # Call the retrieve function
        query = "Sample query"
        serialized, retrieved_docs = retrieve(query)

        # Check that the result contains the expected keys
        assert isinstance(serialized, str)
        assert isinstance(retrieved_docs, list)
        assert len(retrieved_docs) == 3
        assert isinstance(retrieved_docs[0], Document)

        # Check the content of the serialized string
        expected_serialized = (
            "Source: {'source': 'http://example.com/doc1'}\n"
            "Content: Content of document 1\n\n"
            "Source: {'source': 'http://example.com/doc2'}\n"
            "Content: Content of document 2\n\n"
            "Source: {'source': 'http://example.com/doc3'}\n"
            "Content: Content of document 3"
        )
        assert serialized == expected_serialized

        # Check the content of the retrieved documents
        for doc in retrieved_docs:
            assert "Content of document" in doc.page_content


# Run the test
test_retrieve()
