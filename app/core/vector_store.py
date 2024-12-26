import faiss
import bs4
import pickle 
import numpy as np

from langchain.embeddings import OpenAIEmbeddings
# from langchain_core.vectorstores import LangGraphVectorStore #InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# Create a vector store
# vector_store = LangGraphVectorStore(embeddings)

# Function to load and chunk documents
def load_and_chunk_documents(web_paths: list):
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
    
    # Save the FAISS index and documents
    faiss.write_index(index, "app/vector_storage/vector_store.index")
    with open("app/vector_storage/documents.pkl", "wb") as f:
        pickle.dump(all_splits, f)
    
    print(f"Added {len(all_splits)} documents to the vector store.")
    return all_splits

if __name__ == "__main__":
    web_paths = ["https://petguide.dk/hundefoder-maerker/"]
    load_and_chunk_documents(web_paths)