import numpy as np
import faiss
from unittest.mock import patch
from app.core.rag_graph import retrieve
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


def build_mock_vector_store():

    # Create mock documents
    mock_documents = [
        Document(metadata={"source": "http://example.com/doc1"},
                 page_content="Content of document 1"),
        Document(metadata={"source": "http://example.com/doc2"},
                 page_content="Content of document 2")
    ]

    # Initiate the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )

    # Split the documents into chunks
    all_splits = []
    for doc in mock_documents:
        splits = text_splitter.split_documents([doc])
        all_splits.extend(splits)  # Add the splits to the all_splits list

    # Initialize embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Create embeddings for each chunk
    embeddings_list = [embeddings.embed_query(doc.page_content)
                       for doc in all_splits]
    embeddings_array = np.array(embeddings_list)

    # Create a FAISS index
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)

    return index, mock_documents


@patch(
        "app.core.rag_graph.index",
        new_callable=lambda: build_mock_vector_store()[0]
        )
@patch(
        "app.core.rag_graph.documents",
        new_callable=lambda: build_mock_vector_store()[1]
        )
def test_retrieve(mock_documents, mock_index):
    # Call the retrieve function
    query = "retrieve Content of document"
    result = retrieve(query)

    # Check that the result is a string
    assert isinstance(result, str)

    # Check that the result contains two "Source: {'source': ...}" objects
    sources = result.split("Source: ")
    assert len(sources) == 3

    # Check the content of the serialized string
    expected_serialized_part1 = 'http://example.com/doc1'
    expected_serialized_part2 = 'http://example.com/doc2'
    assert expected_serialized_part1 in result
    assert expected_serialized_part2 in result


# Run the test
test_retrieve()
