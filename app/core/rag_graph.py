from typing import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
# from app.core.vector_store import vector_store
from app.core.config import custom_rag_prompt, llm
import pickle 
import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Load the FAISS index and documents
index = faiss.read_index("app/vector_storage/vector_store.index")
with open("app/vector_storage/documents.pkl", "rb") as f:
    documents = pickle.load(f)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    # Create embeddings for the query
    query_embedding = embeddings.embed_query(state["question"])
    query_embedding = np.array([query_embedding])
    
    # Perform similarity search
    distances, indices = index.search(query_embedding, k=5)
    retrieved_docs = [documents[i] for i in indices[0]]
    print(f"Retrieved {len(retrieved_docs)} documents for the question: {state['question']}")
    return {"context": retrieved_docs}

# def retrieve(state: State):
#     retrieved_docs = vector_store.similarity_search(state["question"])
#     print(f"Retrieved {len(retrieved_docs)} documents for the question: {state['question']}")
#     return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = custom_rag_prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content, "source": state["context"]}

# def generate(state: State):
#     docs_content = "\n\n".join(doc.page_content for doc in state["context"])
#     messages = custom_rag_prompt.invoke({"question": state["question"], "context": docs_content})
#     response = llm.invoke(messages)
#     return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()