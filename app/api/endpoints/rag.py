from fastapi import APIRouter
from pydantic import BaseModel
from app.services.rag_service import get_rag_answer

# create a new APIRouter instance
router = APIRouter()

# create a Pydantic model for the question
class Question(BaseModel):
    question: str

# create a POST route for the /ask endpoint
@router.post("/ask")
def ask_question(question: Question):
    result = get_rag_answer(question.question)
    return {"context": result["context"], "answer": result["answer"]}