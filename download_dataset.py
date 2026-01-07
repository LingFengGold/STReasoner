"""
Download ST-Bench dataset from HuggingFace Hub.

Usage:
    python download_dataset.py
"""

from huggingface_hub import snapshot_download
import os

def main():
    # Download ST-Bench dataset to data/ directory
    local_dir = os.path.join(os.path.dirname(__file__), "data", "ST-Bench")
    
    print(f"Downloading ST-Bench dataset to {local_dir}...")
    
    snapshot_download(
        repo_id="Time-HD-Anonymous/ST-Bench",
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # Download actual files, not symlinks
    )
    
    print(f"Download completed! Dataset saved to: {local_dir}")

if __name__ == "__main__":
    main()

