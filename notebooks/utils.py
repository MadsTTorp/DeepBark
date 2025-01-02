import requests
import bs4
from typing import List
import numpy as np
import faiss

from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
llm = ChatOpenAI(model="gpt-4o-mini")

def get_article_links(main_url: str) -> List[str]:
    response = requests.get(main_url)
    soup = bs4.BeautifulSoup(response.content, 'html.parser')
    
    article_links = set(
        [
            a['href'] 
            for a in soup.find_all('a', class_='plain', href=True) 
            if 'https://petguide.dk' in a['href'] and 'kat' not in a['href']
        ]
    )
    
    return list(article_links)

# Function to load and chunk documents
def load_and_chunk_documents(web_paths: List[str]):
    loader = WebBaseLoader(
        web_paths=web_paths,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("entry-content single-page", "entry-title", "entry-meta uppercase is-xsmall")
            )
        ),
    )
    docs = loader.load()

    # Initiate the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # chunk size (characters)
        chunk_overlap=200,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    # Split the documents into chunks
    all_splits = text_splitter.split_documents(docs)
    print(f"Split documents into {len(all_splits)} sub-documents.")
    
    # Create embeddings for each chunk
    embeddings_list = [embeddings.embed_query(doc.page_content) for doc in all_splits]
    embeddings_array = np.array(embeddings_list)
    
    # Create a FAISS index
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
  
    print(f"Added {len(all_splits)} documents to the vector store.")
    return all_splits, index

prompt = hub.pull("rlm/rag-prompt")

prompt_template = """Brug følgende stykker kontekst til at besvare spørgsmålet i slutningen. 
Hvis du ikke kender svaret, så sig bare, at du ikke ved det, og prøv ikke at opdigte et svar.
Svar med maksimalt tre sætninger og hold svaret så kortfattet men præcist som muligt.
Vær høflig i dit svar.

{context} 

Spørgsmål: {question} 

Hjælpsomt svar:"""