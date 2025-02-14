#!/usr/bin/env python3
import os
import argparse
import requests
import logging
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager

# Import configuration defaults.
from src.config import config

# Constants for the DKK website
BASE_URL = "https://www.dkk.dk/race"
RACE_SELECT_CLASS = "lex-custom-select font-semibold pl-2 md: p-1"
RACE_SPEC_CLASS = "race-spec"
LEXICON_CLASS = "md:grid grid-cols-2 gap-x-5"
LEXICON_TEXT_CLASS = "lex-text"
DOCUMENTS_CONTAINER_CLASS = "mx-auto lg:max-w-screen-lg px-10 py-10 lg:py-20"
REQUEST_TIMEOUT = 10
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/91.0.4472.124 Safari/537.36"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Use local cache for webdriver.
os.environ["WDM_LOCAL"] = "1"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Scrape dog breed information from DKK and save as Parquet."
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=config.OUTPUT_PATH,
        help="Directory path for output files"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=config.SCRAPE_OUTPUT_FILE,
        help="Output file name for scraped data"
    )
    return parser.parse_args()

class WebDriverContext:
    """Context manager for handling WebDriver resources."""
    def __enter__(self):
        os.environ["WDM_LOCAL"] = "1"  # Use local driver cache
        options = Options()
        options.add_argument("--headless")
        self.driver = webdriver.Firefox(
            service=Service(GeckoDriverManager().install()),
            options=options
        )
        return self.driver

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.driver.quit()

def get_race_links(driver: webdriver.Firefox, url: str) -> List[str]:
    """Extract race links from the main breed page."""
    try:
        driver.get(url)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        select_element = soup.find("select", class_=RACE_SELECT_CLASS)
        if not select_element:
            logging.warning("No race select element found on page")
            return []
        return [
            option["value"] for option in select_element.find_all("option")
            if option.get("value") and "www.dkk.dk/race" in option["value"]
        ]
    except Exception as e:
        logging.error(f"Error fetching race links: {str(e)}")
        return []

def parse_race_spec(soup: BeautifulSoup) -> Dict[str, str]:
    """Parse race specifications from the detail page."""
    specs = {}
    for spec in soup.find_all("div", class_=RACE_SPEC_CLASS):
        key_element = spec.find("p")
        value_element = spec.find("span")
        if key_element and value_element:
            specs[key_element.get_text(strip=True)] = value_element.get_text(strip=True)
    return specs

def parse_lexicon(soup: BeautifulSoup) -> str:
    """Parse and merge lexicon text sections."""
    lexicon_container = soup.find("div", class_=LEXICON_CLASS)
    if not lexicon_container:
        return ""
    sections = []
    for text_div in lexicon_container.find_all("div", class_=LEXICON_TEXT_CLASS):
        strong = text_div.find("strong")
        if strong:
            title = strong.get_text(strip=True)
            content = text_div.get_text(separator=" ", strip=True).replace(title, "").strip()
            sections.append(f"{title}\n{content}")
    return "\n\n".join(sections)

def parse_documents(soup: BeautifulSoup) -> List[str]:
    """Extract document links from the page."""
    doc_container = soup.find("div", class_=DOCUMENTS_CONTAINER_CLASS)
    return [a["href"] for a in doc_container.find_all("a")] if doc_container else []

def get_dog_info(url: str, session: requests.Session) -> Optional[Dict]:
    """Fetch and parse detailed dog information from an individual race page."""
    try:
        response = session.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Request failed for {url}: {str(e)}")
        return None
    try:
        soup = BeautifulSoup(response.text, "html.parser")
        return {
            "url": url,
            "specs": parse_race_spec(soup),
            "lexicon": parse_lexicon(soup),
            "documents": parse_documents(soup)
        }
    except Exception as e:
        logging.error(f"Parsing failed for {url}: {str(e)}")
        return None

def save_as_parquet(data: List[Dict], filename: str):
    """Save scraped data as a Parquet file."""
    df = pd.DataFrame(data)
    df['scrape_timestamp'] = datetime.now()
    df.to_parquet(filename, engine='pyarrow')
    logging.info(f"Saved scraped data to {filename}")

def main():
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    output_filepath = os.path.join(args.output_path, args.output_file)
    logging.info(f"Output will be saved to: {output_filepath}")

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    with WebDriverContext() as driver:
        race_links = get_race_links(driver, BASE_URL)
        logging.info(f"Found {len(race_links)} race links")

    scraped_data = []
    for link in race_links[:10]:
        data = get_dog_info(link, session)
        if data:
            scraped_data.append(data)
            logging.info(f"Processed: {link}")

    logging.info(f"Successfully collected {len(scraped_data)} dog profiles")
    save_as_parquet(scraped_data, output_filepath)

if __name__ == "__main__":
    main()
