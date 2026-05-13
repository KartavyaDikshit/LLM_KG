import os
import requests
from tqdm import tqdm

def download_file(file_id, dest_path):
    """Download a file from Harvard Dataverse given its file ID."""
    url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    with open(dest_path, 'wb') as f, tqdm(
        desc=os.path.basename(dest_path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = f.write(data)
            bar.update(size)

def fetch_clinvec_data(ontology_dir="data/ontology"):
    """Fetch ClinGraph nodes and edges from Harvard Dataverse."""
    files = {
        "11296211": "ClinGraph_nodes.csv",
        "11296208": "ClinGraph_edges.csv"
    }
    
    print(f"Starting download of ClinVec data to {ontology_dir}...")
    for file_id, filename in files.items():
        dest = os.path.join(ontology_dir, filename)
        if os.path.exists(dest):
            print(f"File {filename} already exists. Skipping.")
            continue
        try:
            download_file(file_id, dest)
            print(f"Successfully downloaded {filename}.")
        except Exception as e:
            print(f"Failed to download {filename}: {e}")

if __name__ == "__main__":
    fetch_clinvec_data()
