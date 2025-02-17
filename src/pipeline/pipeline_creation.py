import os
import argparse
import subprocess
import yaml
from prefect import flow, task, get_run_logger
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
# Import configuration defaults.
from src.config import config

def load_config(config_path: str) -> dict:
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

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
def run_document_creation(scrape_output_path: str, scrape_output_file: str,
                          document_output_path: str, document_output_file: str):
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
def run_index_creation(document_output_path: str, document_output_file: str,
                       index_output_path: str, openai_api_key: str):
    logger = get_run_logger()
    cmd = [
        "python",
        "src/pipeline/generate_index.py",
        "--input-path", document_output_path,
        "--input-file", document_output_file,
        "--output-path", index_output_path,
        "--openai-api-key", openai_api_key
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
    run_document_creation(scrape_output_path, scrape_output_file,
                          document_output_path, document_output_file)
    run_index_creation(document_output_path, document_output_file, index_output_path)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Orchestrate the dog breed data pipeline using Prefect with production config."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/config/production_config.yaml",
        help="Path to the production configuration YAML file."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    config = load_config(args.config)
    # Extract values from the configuration file.
    # We allow overriding a generic "output_path" if more specific keys are not provided.
    scrape_output_path = config.get("scrape_output_path", config.get("output_path", "output"))
    scrape_output_file = config.get("scrape_output_file", "scraped_breeds.parquet")
    document_output_path = config.get("document_output_path", config.get("output_path", "output"))
    document_output_file = config.get("document_output_file", "breed_documents.parquet")
    index_output_path = config.get("index_output_path", config.get("output_path", "output"))

    print("Using configuration:")
    print(f"  Scrape output path:   {scrape_output_path}")
    print(f"  Scrape output file:   {scrape_output_file}")
    print(f"  Document output path: {document_output_path}")
    print(f"  Document output file: {document_output_file}")
    print(f"  Index output path:    {index_output_path}")

    dog_breed_pipeline(scrape_output_path,
                       scrape_output_file,
                       document_output_path,
                       document_output_file,
                       index_output_path)

if __name__ == "__main__":
    main()