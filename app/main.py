from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from app.api.endpoints import rag

# create a FastAPI instance
app = FastAPI()

# include the router from the endpoints
app.include_router(rag.router, prefix="/rag", tags=["rag"])

# serve the static HTML file
app.mount("/", StaticFiles(directory="app/frontend", html=True), name="static")

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG FastAPI application!"}