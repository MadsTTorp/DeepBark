from fastapi import APIRouter
from pydantic import BaseModel
from app.services.rag_service import get_rag_answer

# Create a new APIRouter instance
router = APIRouter()

# Create a Pydantic model for the question
class Question(BaseModel):
    question: str

# Create a POST route for the /ask endpoint
@router.post("/ask")
def ask_question(question: Question):
    result = get_rag_answer(question.question)
    print(result)
    return {"answer": result["answer"], "sources": result["sources"]}
