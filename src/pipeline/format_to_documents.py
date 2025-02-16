import os
import sys
import argparse
import logging
from typing import List, Dict, Any
import pandas as pd
from pydantic import BaseModel
from langchain.schema import Document
import json

# Import configuration defaults.
from src.config import config

class BreedDocument(BaseModel):
    page_content: str
    metadata: Dict[str, Any]

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create document representations from scraped dog breed data."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=config.OUTPUT_PATH,
        help="Directory path where the scraped data file is located."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=config.SCRAPE_OUTPUT_FILE,
        help="Filename of the scraped data (Parquet format)."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=config.OUTPUT_PATH,
        help="Directory path where the documents file will be saved."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=config.DOCUMENT_OUTPUT_FILE,
        help="Output filename for the documents (Parquet format)."
    )
    return parser.parse_args()


def load_scraped_data(filename: str) -> pd.DataFrame:
    try:
        df = pd.read_parquet(filename)
        logging.info(f"Loaded data from {filename}")
        return df
    except Exception as e:
        logging.error(f"Failed to load data from {filename}: {e}")
        raise

def format_specs(specs: Dict) -> str:
    return "\n".join([f"- {k}: {v}" for k, v in specs.items()])

def format_content(row: pd.Series) -> str:
    breed_name = row["url"].rstrip("/").split("/")[-1].capitalize()
    content = [
        f"Breed Profile: {breed_name}",
        "## Key Characteristics:",
        format_specs(row["specs"]),
        "## Detailed Description:",
        row["lexicon"]
    ]
    return "\n".join(content)

def create_documents(df: pd.DataFrame) -> List[Document]:
    documents = []
    for _, row in df.iterrows():
        breed_name = row["url"].rstrip("/").split("/")[-1]
        doc = Document(
            page_content=format_content(row),
            metadata={
                "source": row["url"],
                "specs": row["specs"],
                "documents": row["documents"],
                "breed_name": breed_name,
                "scrape_timestamp": row["scrape_timestamp"],
                "content_type": "dog_breed_profile"
            }
        )
        documents.append(doc)
    logging.info(f"Created {len(documents)} documents.")
    return documents

def save_documents(documents: list[Document], filename: str):
    try:
        docs = []
        for doc in documents:
            # Convert the Document to a dictionary.
            doc_dict = doc.dict()
            # Convert the metadata dict to a JSON string.
            doc_dict["metadata"] = json.dumps(doc_dict["metadata"])
            docs.append(doc_dict)
        df = pd.DataFrame(docs)
        df.to_parquet(filename, engine="pyarrow")
        logging.info(f"Documents saved to {filename}")
    except Exception as e:
        logging.error(f"Failed to save documents to {filename}: {e}")
        raise

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )
    args = parse_args()
    input_filepath = os.path.join(args.input_path, args.input_file)
    
    # Load scraped data and exit if empty
    df = load_scraped_data(input_filepath)
    if df.empty:
        logging.error("Scraped data is empty. Exiting without overwriting document file.")
        sys.exit(1)
    
    logging.info("Creating documents from scraped data.")
    documents = create_documents(df)
    if not documents:
        logging.error("No documents were created. Exiting pipeline.")
        sys.exit(1)
    
    os.makedirs(args.output_path, exist_ok=True)
    output_filepath = os.path.join(args.output_path, args.output_file)
    logging.info(f"Saving documents to {output_filepath}")
    save_documents(documents, output_filepath)

if __name__ == "__main__":
    main()