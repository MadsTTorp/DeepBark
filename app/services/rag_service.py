from app.core.init_rag import get_graph


def get_rag_answer(question: str) -> dict:
    graph = get_graph()
    print(f"Received question: {question}")
    result = graph.invoke({"question": question})
    print(
        f"Answer: {result['answer']['answer']},\
        Sources: {result['answer']['sources']}"
    )

    return {"context": result["context"], "answer": result["answer"]}
