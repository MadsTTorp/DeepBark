#!/usr/bin/env python3
import os
import argparse
import logging
import pandas as pd
import numpy as np
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from langchain.schema import Document
from sentence_transformers import SentenceTransformer

# Import configuration defaults.
from src.config import config

# Define a custom cache directory for storing the model.
MODEL_CACHE_DIR = os.path.expanduser("~/.cache/sentence_transformers/")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

MODEL_NAME = "all-MiniLM-L6-v2"
logging.info(f"Loading model '{MODEL_NAME}' from cache directory: {MODEL_CACHE_DIR}")
model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_CACHE_DIR)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a FAISS vector search index from document data."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=config.OUTPUT_PATH,
        help="Directory path where the document data file is located."
    )
    parser.add_argument(
        "--input-file", type=str, default=config.DOCUMENT_OUTPUT_FILE,
        help="Path to input documents file (Parquet format)."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=config.OUTPUT_PATH,
        help="Directory where the FAISS index and chunk metadata will be stored."
    )
    parser.add_argument(
        "--chunk-size", type=int, default=1000,
        help="Chunk size in characters for splitting documents."
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=200,
        help="Overlap (in characters) between chunks."
    )
    parser.add_argument(
        "--model-name", type=str, default="all-MiniLM-L6-v2",
        help="SentenceTransformer model name for embedding generation."
    )
    return parser.parse_args()

def load_documents(filename: str) -> list[Document]:
    try:
        with open(filename, "rb") as f:
            documents = pickle.load(f)
        logging.info(f"Loaded {len(documents)} documents from {filename}")
        return documents
    except Exception as e:
        logging.error(f"Error loading documents from {filename}: {e}")
        raise

def chunk_documents(documents, chunk_size=1000, chunk_overlap=200):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    all_chunks = []
    for doc in documents:
        chunks = text_splitter.split_text(doc["page_content"])
        for chunk in chunks:
            all_chunks.append({
                "page_content": chunk,
                "metadata": doc["metadata"]
            })
    logging.info(f"Created {len(all_chunks)} chunks from documents.")
    return all_chunks

def create_index(chunks):
    embeddings = [model.encode(chunk["page_content"], convert_to_numpy=True) for chunk in chunks]
    embeddings_array = np.vstack(embeddings)
    
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array)
    
    logging.info(f"FAISS index created with {index.ntotal} vectors (dimension {dimension}).")
    return index

def save_index(index, output_index_filepath):
    try:
        faiss.write_index(index, output_index_filepath)
        logging.info(f"Index saved to {output_index_filepath}.")
    except Exception as e:
        logging.error(f"Error saving index: {e}")
        raise

def save_chunks(chunks, output_chunks_filepath):
    try:
        with open(output_chunks_filepath, "wb") as f:
            pickle.dump(chunks, f)
        logging.info(f"Chunk metadata saved to {output_chunks_filepath}.")
    except Exception as e:
        logging.error(f"Error saving chunk metadata: {e}")
        raise

def main():
    args = parse_args()
    input_filepath = os.path.join(args.input_path, args.input_file)
    os.makedirs(args.output_path, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    documents = load_documents(input_filepath)
    chunks = chunk_documents(
        documents, 
        chunk_size=args.chunk_size, 
        chunk_overlap=args.chunk_overlap
    )
    index = create_index(chunks)
    index_filepath = os.path.join(args.output_path, config.INDEX_FILE)
    chunks_filepath = os.path.join(args.output_path, config.CHUNKS_FILE)
    save_index(index, index_filepath)
    save_chunks(chunks, chunks_filepath)
    logging.info("Indexing process completed successfully.")

if __name__ == "__main__":
    main()
