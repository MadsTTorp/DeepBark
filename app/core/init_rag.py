from app.core.rag_graph import graph


def get_graph():
    return graph


if __name__ == "__main__":
    graph = get_graph()
    print("RAG model initialized.")
