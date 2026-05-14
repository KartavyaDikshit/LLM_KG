import pandas as pd
import os

def load_clinical_notes(file_path):
    """
    Load clinical notes from a CSV file.
    """
    # Try multiple path variations for robustness
    paths_to_try = [
        file_path,
        os.path.join(os.getcwd(), file_path),
        os.path.join("/content/LLM_KG", file_path),
        os.path.join("datasets", os.path.basename(file_path))
    ]
    
    found_path = None
    for p in paths_to_try:
        if os.path.exists(p):
            found_path = p
            break
            
    if not found_path:
        raise FileNotFoundError(f"Clinical notes file not found after checking: {paths_to_try}")
    
    print(f"Loading clinical notes from {found_path}...")
    df = pd.read_csv(found_path)
    
    # Identify the text column
    possible_text_cols = ['text', 'note', 'description', 'NEV_TEXT', 'CONTENT']
    text_col = next((col for col in possible_text_cols if col in df.columns), None)
    
    if not text_col:
        raise ValueError(f"Could not find text column in {df.columns}. Expected one of {possible_text_cols}")
    
    return df[text_col].dropna().tolist()

def load_ontology(ontology_dir="data/ontology"):
    """
    Load ClinVec ontology nodes and edges.
    Checks both data/ontology and datasets/ontology.
    """
    paths_to_check = [ontology_dir, os.path.join("datasets", "ontology"), "datasets"]
    
    nodes_df = None
    edges_df = None
    
    for base in paths_to_check:
        n_path = os.path.join(base, "ClinGraph_nodes.csv")
        e_path = os.path.join(base, "ClinGraph_edges.csv")
        if os.path.exists(n_path) and os.path.exists(e_path):
            print(f"Loading ontology data from {base}...")
            nodes_df = pd.read_csv(n_path)
            edges_df = pd.read_csv(e_path)
            break
            
    if nodes_df is None:
        print("Ontology files missing in data/ or datasets/. Please run fetcher.py or check folder structure.")
    
    return nodes_df, edges_df
