import os
import requests
import zipfile
import subprocess
from dotenv import load_dotenv 
load_dotenv()

def get_repo_root():
    """
    Returns the repository's root directory by running a git command.
    Falls back to the current working directory if git is not available.
    """
    try:
        repo_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.STDOUT
        ).strip().decode("utf-8")
        return repo_root
    except Exception as e:
        print("Could not determine repository root using git. Using current working directory.")
        return os.getcwd()

def download_latest_release(owner, repo, dest_dir, github_token=None):
    # GitHub API endpoint for the latest release
    api_url = f"https://api.github.com/repos/{owner}/{repo}/releases/latest"
    
    headers = {}
    if github_token:
        headers["Authorization"] = f"token {github_token}"
    
    print(f"Fetching latest release info from {api_url} ...")
    response = requests.get(api_url, headers=headers)
    response.raise_for_status()
    release_info = response.json()

    tag_name = release_info.get("tag_name", "Unknown")
    print("Latest release tag:", tag_name)
    
    assets = release_info.get("assets", [])
    if not assets:
        print("No assets found in the latest release.")
        return
    
    # For simplicity, use the first asset; adjust this if you have multiple and need to filter by name.
    asset = assets[0]
    asset_name = asset.get("name")
    download_url = asset.get("browser_download_url")
    
    if not download_url:
        print("No download URL found for the asset.")
        return

    print(f"Downloading asset '{asset_name}' from {download_url} ...")
    asset_response = requests.get(download_url, headers=headers)
    asset_response.raise_for_status()
    
    asset_path = os.path.join(dest_dir, asset_name)
    with open(asset_path, "wb") as f:
        f.write(asset_response.content)
    print("Asset downloaded to:", asset_path)
    
    # If the asset is a zip file, extract its contents to dest_dir.
    if asset_name.lower().endswith(".zip"):
        print("Extracting zip file...")
        with zipfile.ZipFile(asset_path, "r") as zip_ref:
            zip_ref.extractall(dest_dir)
        print("Extraction complete.")
        os.remove(asset_path)
    else:
        print("Asset is not a zip file; skipping extraction.")


if __name__ == "__main__":
    # Replace these with your repository details
    owner = os.getenv('REPO_USERNAME')        # e.g., "DeepBarkOrg" or your username
    repo = os.getenv('REPO_NAME')      # e.g., "DeepBark"

    repo_root = get_repo_root()
    dest_dir = os.path.join(repo_root, "app", "vector_storage")
    os.makedirs(dest_dir, exist_ok=True)
    os.makedirs(dest_dir, exist_ok=True)
    
    # Optionally, set a GitHub token in your environment to avoid rate limits
    github_token = os.getenv("GITHUB_TOKEN")
    
    download_latest_release(owner, repo, dest_dir, github_token)
