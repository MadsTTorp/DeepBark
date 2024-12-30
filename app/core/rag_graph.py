import pickle
import faiss
import numpy as np
from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import START, StateGraph
from app.core.config import custom_rag_prompt, llm
from typing_extensions import Annotated, TypedDict

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Load the FAISS index and documents
index = faiss.read_index("app/vector_storage/vector_store.index")
with open("app/vector_storage/documents.pkl", "rb") as f:
    documents = pickle.load(f)

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


class AnswerWithSources(TypedDict):
    """An answer to the question, with sources."""

    answer: str
    sources: Annotated[
        List[str],
        ...,
        "List of sources (author + year) used to answer the question",
    ]


class State(TypedDict):
    question: str
    context: List[Document]
    answer: AnswerWithSources


def retrieve(state: State):
    # create embeddings for the query
    query_embedding = embeddings.embed_query(state["question"])
    query_embedding = np.array([query_embedding])
    # perform similarity search
    k = 3
    similarity_threshold = 0.7
    distances, indices = index.search(query_embedding, k=k)
    # filter documents based on the similarity threshold
    retrieved_docs = []
    for distance, idx in zip(distances[0], indices[0]):
        if distance < similarity_threshold:
            retrieved_docs.append(documents[idx])
    print(retrieved_docs)
    state = {"context": retrieved_docs}
    
    return state


def generate(state: State):
    if not state["context"]:
        response = {'answer': 'Jeg kender desværre ikke svaret på dit '
                              'spørgsmål, på baggrund af de artikler '
                              'jeg har adgang til.',
                    'sources': []}
        return {"answer": response}
        
    else:
        # concatenate the content of the retrieved documents
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        # invoke the custom RAG prompt
        messages = custom_rag_prompt.invoke(
            {"question": state["question"], "context": docs_content}
        )
        # invoke the LLM with structured output
        structured_llm = llm.with_structured_output(AnswerWithSources)
        # invoke the LLM with the messages
        response = structured_llm.invoke(messages)
        # Extract unique URLs from the context
        unique_urls = list({doc.metadata["source"] for doc in state["context"]})
        # Update the response with the unique URLs
        response["sources"] = unique_urls

        return {"answer": response}


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
