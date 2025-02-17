import pickle
import faiss
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langgraph.prebuilt import ToolNode, tools_condition
from app.core.config import llm, get_prompt, similarity_threshold
from langgraph.graph import MessagesState, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Load the FAISS index and documents
index = faiss.read_index("app/vector_storage/faiss_index.index")
with open("app/vector_storage/chunked_documents.pkl", "rb") as f:
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
        if distance < 1.0:
            doc = documents[idx]
            # Convert to a dictionary format similar to LangChain's Document class
            retrieved_docs.append({
                "metadata": doc.metadata,
                "page_content": doc.page_content
            })
    serialized = "\n\n".join(
        (f"Source: {doc['metadata']['source']}\n" f"Content: {doc['page_content']}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs


# Generate an AIMessage that may include a tool-call to be sent.
def query_or_respond(state: MessagesState):
    """Generate tool call for retrieval or respond."""
    llm_with_tools = llm.bind_tools([retrieve])
    response = llm_with_tools.invoke(state["messages"])
    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}


# Executre the retrieval tool
tools = ToolNode([retrieve])


# Generate a response using the retrieved content.
def generate(state: MessagesState):
    """Generate answer."""

    # collect the last tool message (contains sources)
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    # Format into prompt
    docs_content = "\n\n".join(doc.content for doc in tool_messages)
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
# add conditions for passing from one node to another
graph_builder.add_conditional_edges(
    "query_or_respond",
    tools_condition,
    {END: END, "tools": "tools"},
)
# add edges to the graph that connect the nodes
graph_builder.add_edge("tools", "generate")
graph_builder.add_edge("generate", END)
# initiate the memory saver to save the state of the graph
graph = graph_builder.compile(checkpointer=memory)
