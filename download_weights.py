import os
from huggingface_hub import HfApi, login, hf_hub_download

# Define file paths and repository details
files_to_download = [
    "esm3_function_decoder_v0.pth",
    "esm3_sm_open_v1.pth",
    "esm3_structure_decoder_v0.pth",
    "esm3_structure_encoder_v0.pth"
]

weights_dir = "./data/weights"
repo_id = "EvolutionaryScale/esm3-sm-open-v1"

# Ensure the weights directory exists
os.makedirs(weights_dir, exist_ok=True)

# Log in to Hugging Face (will ask for your token the first time)
login()

# Download files if they do not already exist
api = HfApi()

for file_name in files_to_download:
    file_path = os.path.join(weights_dir, file_name)
    
    if not os.path.exists(file_path):
        print(f"Downloading {file_name}...")
        try:
            # Download the file from the Hugging Face repository
            hf_hub_download(repo_id=repo_id, filename=file_path, local_dir=weights_dir)
            print(f"Downloaded {file_name} successfully.")
        except Exception as e:
            print(f"Failed to download {file_name}: {e}")
    else:
        print(f"{file_name} already exists, skipping download.")
