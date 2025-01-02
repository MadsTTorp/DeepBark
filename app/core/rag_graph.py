import pickle
import faiss
import numpy as np
from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langgraph.graph import START, StateGraph
from app.core.config import *
from typing_extensions import Annotated, TypedDict
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_core.tools import tool

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Load the FAISS index and documents
index = faiss.read_index("app/vector_storage/vector_store.index")
with open("app/vector_storage/documents.pkl", "rb") as f:
    documents = pickle.load(f)

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Initialize the graph builder
graph_builder = StateGraph(MessagesState)

# Initialize the memory saver
memory = MemorySaver()


@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    # Create embeddings for the query
    query_embedding = embeddings.embed_query(query)
    query_embedding = np.array([query_embedding])

    distances, indices = index.search(query_embedding, k=3)
    retrieved_docs = []
    for distance, idx in zip(distances[0], indices[0]):
        if distance < similarity_threshold:
            retrieved_docs.append(documents[idx])

    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# Executre the retrieval tool
tools = ToolNode([retrieve])

# Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


#Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""

    # collect the last tool message (contains sources)
    last_tool_message = next(
        (
            tool_msg for tool_msg in reversed(state["messages"])
            if isinstance(tool_msg, ToolMessage)
        )
    )

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in [last_tool_message])
    system_message_cont = get_prompt(docs_content)
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]
    prompt = [SystemMessage(system_message_cont)] + conversation_messages

    # Run
    response = llm.invoke(prompt)
    return {"messages": [response]}

# build the graph
graph_builder.add_node(query_or_respond)
graph_builder.add_node(tools)
graph_builder.add_node(generate)

# set the entry point
graph_builder.set_entry_point("query_or_respond")
# add edges to the graph that connect the nodes
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
# add edges to the graph that connect the nodes
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)
# compile the graph
graph = graph_builder.compile()
# initiate the memory saver to save the state of the graph
graph = graph_builder.compile(checkpointer=memory)
