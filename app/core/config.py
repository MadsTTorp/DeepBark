from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

llm = ChatOpenAI(model="gpt-4o-mini")

prompt_template = (
    "Brug følgende stykker kontekst til at besvare spørgsmålet i "
    "slutningen. Hvis du ikke kender svaret, så sig bare, at du "
    "ikke ved det, og prøv ikke at opdigte et svar. Svar med "
    "maksimalt tre sætninger og hold svaret så kortfattet men "
    "præcist som muligt. Vær høflig i dit svar.\n\n"
    "{context}\n\nSpørgsmål: {question}\n\nHjælpsomt svar:"
)

custom_rag_prompt = PromptTemplate.from_template(prompt_template)
