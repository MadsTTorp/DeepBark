import argparse
import requests
import bs4
import os
import numpy as np
import pickle
import faiss
import logging
import shutil
from pathlib import Path
from typing import List
from langchain.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# Initialize logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s'
                    )

# Define the command line arguments
def _init_parser():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        prog=__file__,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # REQUIRED ARGUMENTS
    parser.add_argument(
        "--main_url",
        type=str,
        required=True,
        help="The main URL to collect articles from.",
    )
    # OPTIONAL ARGUMENTS
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        required=False,
        help="The chunk size for splitting documents.",
    )
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=200,
        required=False,
        help="The chunk overlap for splitting documents.",
    )
    parser.add_argument(
        "--storage_dir",
        type=str,
        default="app/vector_storage",
        required=False,
        help="The directory to store the vector index and documents.",
    )
    parser.add_argument(
        "--index_name",
        type=str,
        default="vector_store.index",
        required=False,
        help="The name of the FAISS index file.",
    )
    parser.add_argument(
        "--document_name",
        type=str,
        default="documents.pkl",
        required=False,
        help="The name of the document pickle file.",
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        default=False,
        help='Run the script without making any changes.',
    )

    return parser.parse_args()


# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

def get_article_links(main_url: str) -> List[str]:
    try:
        response = requests.get(main_url)
        response.raise_for_status()
        soup = bs4.BeautifulSoup(response.content, 'html.parser')
        
        article_links = set(
            [
                a['href'] 
                for a in soup.find_all('a', class_='plain', href=True) 
                if 'https://petguide.dk' in a['href'] and 'kat' not in a['href']
            ]
        )
        
        return list(article_links)
    except requests.RequestException as e:
        logging.error(f"Error fetching article links: {e}")
        return []

def load_and_chunk_documents(web_paths: List[str], 
                             chunk_size: int, 
                             chunk_overlap: int, 
                             storage_dir: str, 
                             index_name: str, 
                             document_name: str,
                             dry_run: bool = False) -> List:
    try:
        loader = WebBaseLoader(
            web_paths=web_paths,
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("entry-content single-page", 
                            "entry-title", 
                            "entry-meta uppercase is-xsmall")
                )
            ),
        )
        docs = loader.load()

        # Initiate the text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,  # chunk size (characters)
            chunk_overlap=chunk_overlap,  # chunk overlap (characters)
            add_start_index=True,  # track index in original document
        )
        # Split the documents into chunks
        all_splits = text_splitter.split_documents(docs)
        logging.info(f"Splitting documents into {len(all_splits)} sub-documents...")
        
        if dry_run:
            logging.info("Dry run enabled. Exiting without saving.")
            return all_splits
        
        # Create embeddings for each chunk
        embeddings_list = [embeddings.embed_query(doc.page_content) for doc in all_splits]
        embeddings_array = np.array(embeddings_list)
        
        # Create a FAISS index
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array)

        # Backup existing index and documents
        if os.path.exists(f"{storage_dir}/{index_name}"):
            shutil.copy(f"{storage_dir}/{index_name}", f"{storage_dir}/{index_name}.bak")
        if os.path.exists(f"{storage_dir}/{document_name}"):
            shutil.copy(f"{storage_dir}/{document_name}", f"{storage_dir}/{document_name}.bak")

        # Save the FAISS index and documents
        faiss.write_index(index, f"{storage_dir}/{index_name}")
        with open(f"{storage_dir}/{document_name}", "wb") as f:
            pickle.dump(all_splits, f)
        
        logging.info(f"Added {len(all_splits)} documents to the vector store...")
        return all_splits
    except Exception as e:
        logging.error(f"Error during document loading and chunking: {e}")
        return []

def main():

    args = _init_parser()
    Path(args.storage_dir).mkdir(parents=True, exist_ok=True)
    
    logging.info(f'Getting article links from {args.main_url}...')
    article_links = get_article_links(args.main_url)
    logging.info(f'Found {len(article_links)} article links...')
    
    if article_links:
        load_and_chunk_documents(article_links, 
                                 args.chunk_size, 
                                 args.chunk_overlap, 
                                 args.storage_dir, 
                                 args.index_name, 
                                 args.document_name,
                                 args.dry_run)
        logging.info(f'Finished loading and chunking documents. \
              Vector store is stored in {args.storage_dir}/')
    else:
        logging.info('No article links found. Exiting.')

if __name__ == "__main__":
    main()