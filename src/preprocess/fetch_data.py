#!/usr/bin/env python3

import requests
import sqlite3
import pandas as pd
from pathlib import Path
import logging
import tempfile

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

OUTPUT_DIR = Path("/Users/callumanderson/Library/Mobile Documents/com~apple~CloudDocs/Documents/Documents - Callum’s Laptop/Masters-File-Repo/pytorch-learning/pricepoke/data/raw")
GITHUB_REPO = "poketrax/pokedata"
API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"

FILE_MAPPING = {
    "data.sqlite": {
        "table_name": "cards",
        "csv_name": "pokemon_data.csv"
    },
    "prices.sqlite": {
        "table_name": "prices",
        "csv_name": "pokemon_prices.csv"
    }
}

def get_latest_release_assets():
    logging.info(f"Fetching latest release information from {API_URL}")
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        release_data = response.json()

        asset_urls = {}
        for asset in release_data.get("assets", []):
            asset_name = asset.get("name")
            if asset_name in FILE_MAPPING:
                asset_urls[asset_name] = asset.get("browser_download_url")

        missing_files = set(FILE_MAPPING.keys()) - set(asset_urls.keys())
        if missing_files:
            logging.error(f"Could not find the following assets in the latest release: {', '.join(missing_files)}")
            return None

        return asset_urls
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching release info from GitHub: {e}")
        return None

def download_file(url: str, destination_folder: Path) -> Path | None:
    local_filename = url.split('/')[-1]
    file_path = destination_folder / local_filename
    logging.info(f"Downloading {url} to {file_path}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        logging.info(f"Successfully downloaded {local_filename}")
        return file_path
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading {url}: {e}")
        return None

def convert_sqlite_to_csv(db_path: Path, table_name: str, csv_path: Path):
    logging.info(f"Converting {db_path.name} (table: {table_name}) to {csv_path.name}...")
    try:
        con = sqlite3.connect(db_path)
        
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", con)
        
        con.close()
        
        df.to_csv(csv_path, index=False)
        logging.info(f"Successfully converted and saved to {csv_path}")
    except Exception as e:
        logging.error(f"Error converting {db_path.name} to CSV: {e}")
        if 'con' in locals() and con:
            con.close()

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    asset_urls = get_latest_release_assets()
    if not asset_urls:
        logging.error("Could not retrieve asset URLs. Exiting.")
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Using temporary directory: {temp_dir}")
        temp_path = Path(temp_dir)
        for sqlite_file, details in FILE_MAPPING.items():
            if sqlite_file in asset_urls:
                url = asset_urls[sqlite_file]
                
                downloaded_db_path = download_file(url, temp_path)
                
                if downloaded_db_path:
                    csv_path = OUTPUT_DIR / details["csv_name"]
                    convert_sqlite_to_csv(downloaded_db_path, details["table_name"], csv_path)
            else:
                logging.error(f"{sqlite_file} not found in the latest release assets.")

if __name__ == "__main__":
    main()