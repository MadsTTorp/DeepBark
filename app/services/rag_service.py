from app.core.rag_graph import graph
from app.core.config import memory_config
from langchain_core.messages import ToolMessage
from langchain_core.messages import AIMessage
import logging

logger = logging.getLogger(__name__)


def get_rag_answer(question: str) -> dict:

    logger.info(f"Received question: {question}")
    # try:

    # define query
    query = {"messages": [{"role": "user", "content": question}]}
    # send query to graph
    output = graph.stream(query, stream_mode="values", config=memory_config)

    # Collect the response stream from the graph
    result = [step["messages"] for step in output][-1]

    # logging.info(f"Response stream: {result}")
    logging.info(f"Config: {memory_config}")
    # Extract the final response
    final_response = next(
        (res for res in reversed(result) if isinstance(res, AIMessage)), None
    )
    answer = final_response.content

    # Extract sources from ToolMessages
    tool_message = next(
        (
            tool_msg
            for tool_msg in reversed(result)
            if isinstance(tool_msg, ToolMessage)
        ),
        None,
    )
    try:
        if tool_message:
            source_list = []
            for doc in tool_message.artifact:
                if hasattr(doc, "metadata") and "source" in doc.metadata:
                    source_list.append(doc.metadata["source"])
                elif (
                    isinstance(doc, dict)
                    and "metadata" in doc
                    and "source" in doc["metadata"]
                ):
                    source_list.append(doc["metadata"]["source"])

        else:
            source_list = []

        # Remove duplicates
        source_list = list(set(source_list))

        # default response in case of no sources
        if not source_list:
            answer = (
                "Jeg kender desværre ikke svaret på dit "
                "spørgsmål, på baggrund af de artikler "
                "jeg har adgang til."
            )

        logger.info(f"Answer: {answer}, Sources: {source_list}")

    except Exception as e:
        logger.error(f"Error: {e}")
        answer = "Noget gik galt, prøv venligst igen."
        source_list = []

    return {"answer": answer, "sources": source_list}
