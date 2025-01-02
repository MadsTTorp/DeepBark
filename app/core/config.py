from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import uuid
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Initialize the language model
llm = ChatOpenAI(model="gpt-4o-mini")

# Define the prompt template
def get_prompt(docs_context):
    prompt_template = (
        "Du er en ekspert i alt omkring hunde."
        "Brug udelukkende følgende stykker kontekst til at besvare spørgsmålet. "
        "Svar med maksimalt tre sætninger og hold svaret så kortfattet men "
        "præcist som muligt og velformuleret. Vær høflig i dit svar."
        "\n\n"
        f"{docs_context}"
    )
    return prompt_template

# Generate a unique thread ID for each session
def generate_thread_id():
    return str(uuid.uuid4())

memory_config = {"configurable": {"thread_id": generate_thread_id()}}

similarity_threshold = 1.0