import os
from huggingface_hub import snapshot_download

def download_data():
    print("Downloading datasets from Hugging Face...")
    snapshot_download(repo_id="chemahc94/Dental_003_Dataset", repo_type="dataset", local_dir=".")
    
    print("Downloading model checkpoints from Hugging Face...")
    snapshot_download(repo_id="chemahc94/Dental_003_Model", repo_type="model", local_dir=".")
    
    print("Download completed!")

if __name__ == "__main__":
    download_data()
