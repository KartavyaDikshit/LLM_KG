import pandas as pd
import os

def load_clinical_notes(file_path):
    """
    Load clinical notes from a CSV file.
    Expects a structure similar to MIMIC-IV (e.g., column 'text' or 'description').
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Clinical notes file not found: {file_path}")
    
    print(f"Loading clinical notes from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Identify the text column
    possible_text_cols = ['text', 'note', 'description', 'NEV_TEXT']
    text_col = next((col for col in possible_text_cols if col in df.columns), None)
    
    if not text_col:
        raise ValueError(f"Could not find text column in {df.columns}. Expected one of {possible_text_cols}")
    
    # Return a list of notes
    return df[text_col].dropna().tolist()

def load_ontology(ontology_dir="data/ontology"):
    """
    Load ClinVec ontology nodes and edges.
    """
    nodes_path = os.path.join(ontology_dir, "ClinGraph_nodes.csv")
    edges_path = os.path.join(ontology_dir, "ClinGraph_edges.csv")
    
    if not os.path.exists(nodes_path) or not os.path.exists(edges_path):
        print("Ontology files missing. Please run fetcher.py first.")
        return None, None
    
    print("Loading ontology data (this may take a moment)...")
    nodes_df = pd.read_csv(nodes_path)
    edges_df = pd.read_csv(edges_path)
    
    return nodes_df, edges_df
