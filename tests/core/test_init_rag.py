from app.core.init_rag import get_graph
from app.core.rag_graph import graph


def test_get_graph():
    # Ensure that get_graph returns the correct graph object
    assert get_graph() is graph
