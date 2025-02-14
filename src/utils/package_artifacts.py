import os
import shutil
import argparse
from src.config import config  # Import configuration from the config module

def package_artifacts(output_path, archive_name):
    # Check if the output directory exists
    if not os.path.isdir(output_path):
        print(f"Output directory '{output_path}' does not exist. Aborting packaging step.")
        exit(1)
    
    # Calculate the parent directory of output_path
    parent_dir = os.path.dirname(os.path.abspath(output_path))
    archive_path = os.path.join(parent_dir, archive_name)
    
    # Create the archive in the parent directory so it doesn't get included in the archive itself.
    shutil.make_archive(base_name=archive_path, format="zip", root_dir=output_path)
    print(f"Artifacts packaged at {archive_path}.zip")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Package pipeline artifacts into a zip file."
    )
    parser.add_argument(
        "--output-path", 
        type=str, 
        default=config.OUTPUT_PATH, 
        help="Directory with artifacts."
    )
    parser.add_argument(
        "--archive-name", 
        type=str, 
        default=f"release_artifacts_{config.RELEASE_VERSION}",
        help="Name of the archive."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    package_artifacts(args.output_path, args.archive_name)

if __name__ == "__main__":
    main()
