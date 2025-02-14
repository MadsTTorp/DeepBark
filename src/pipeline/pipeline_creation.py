#!/usr/bin/env python3
import subprocess
import os
from prefect import flow, task, get_run_logger
import argparse

# Import configuration defaults.
from src.config import config

@task
def run_scraping(scrape_output_path: str, scrape_output_file: str):
    logger = get_run_logger()
    cmd = [
        "python",
        "src/pipeline/website_scrapers/dkk_scraper.py",
        "--output-path", scrape_output_path,
        "--output-file", scrape_output_file
    ]
    logger.info(f"Running scraping script: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info("Scraping step completed.")

@task
def run_document_creation(scrape_output_path: str, 
                          scrape_output_file: str,
                          document_output_path: str, 
                          document_output_file: str):
    logger = get_run_logger()
    cmd = [
        "python",
        "src/pipeline/format_to_documents.py",
        "--input-path", scrape_output_path,
        "--input-file", scrape_output_file,
        "--output-path", document_output_path,
        "--output-file", document_output_file
    ]
    logger.info(f"Running document creation script: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info("Document creation step completed.")

@task
def run_index_creation(document_output_path: str, 
                       document_output_file: str,
                       index_output_path: str):
    logger = get_run_logger()
    cmd = [
        "python",
        "src/pipeline/generate_index.py",
        "--input-path", document_output_path,
        "--input-file", document_output_file,
        "--output-path", index_output_path
    ]
    logger.info(f"Running index creation script: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info("Index creation step completed.")

@flow(name="Dog Breed Pipeline")
def dog_breed_pipeline(scrape_output_path: str,
                       scrape_output_file: str,
                       document_output_path: str,
                       document_output_file: str,
                       index_output_path: str):
    os.makedirs(scrape_output_path, exist_ok=True)
    os.makedirs(document_output_path, exist_ok=True)
    os.makedirs(index_output_path, exist_ok=True)
    run_scraping(scrape_output_path, scrape_output_file)
    run_document_creation(
        scrape_output_path, 
        scrape_output_file,
        document_output_path, 
        document_output_file
    )
    run_index_creation(
        document_output_path, 
        document_output_file, 
        index_output_path
    )

def parse_pipeline_args():
    parser = argparse.ArgumentParser(
        description="Orchestrate the dog breed data pipeline using Prefect."
    )
    parser.add_argument(
        "--scrape-output-path",
        type=str,
        default=config.OUTPUT_PATH,
        help="Directory path for the scraping output."
    )
    parser.add_argument(
        "--scrape-output-file",
        type=str,
        default=config.SCRAPE_OUTPUT_FILE,
        help="Filename for the scraping output (Parquet format)."
    )
    parser.add_argument(
        "--document-output-path",
        type=str,
        default=config.OUTPUT_PATH,
        help="Directory path for the document output."
    )
    parser.add_argument(
        "--document-output-file",
        type=str,
        default=config.DOCUMENT_OUTPUT_FILE,
        help="Filename for the document output (Parquet format)."
    )
    parser.add_argument(
        "--index-output-path",
        type=str,
        default=config.OUTPUT_PATH,
        help="Directory path for the index output."
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_pipeline_args()
    dog_breed_pipeline(
        scrape_output_path=args.scrape_output_path,
        scrape_output_file=args.scrape_output_file,
        document_output_path=args.document_output_path,
        document_output_file=args.document_output_file,
        index_output_path=args.index_output_path
    )
