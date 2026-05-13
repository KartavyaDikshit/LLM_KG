import os
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.ingestion.fetcher import fetch_clinvec_data
from src.ingestion.loader import load_clinical_notes
from src.agents.graph import create_agentic_workflow
from src.graph.builder import GraphBuilder
from src.graph.visualizer import visualize_graph

def main():
    parser = argparse.ArgumentParser(description="Personalized Medicine Knowledge Graph Generator")
    parser.add_argument("--notes", type=str, help="Path to MIMIC-IV clinical notes CSV", default="data/raw/notes_sample.csv")
    parser.add_argument("--fetch-ontology", action="store_true", help="Download ClinVec ontology from Dataverse")
    parser.add_argument("--limit", type=int, help="Limit the number of notes to process", default=5)
    
    args = parser.parse_args()

    # Step 1: Fetch Ontology if requested
    if args.fetch_ontology:
        fetch_clinvec_data()

    # Step 2: Load Clinical Notes
    if not os.path.exists(args.notes):
        print(f"Notes file not found: {args.notes}. Please provide a valid CSV or use --fetch-ontology and a sample.")
        return

    notes = load_clinical_notes(args.notes)
    notes = notes[:args.limit]
    print(f"Processing {len(notes)} clinical notes...")

    # Step 3: Initialize Pipeline and Graph
    workflow = create_agentic_workflow()
    builder = GraphBuilder()

    # Step 4: Run Agentic Pipeline for each note
    for i, note in enumerate(notes):
        print(f"--- Processing Note {i+1}/{len(notes)} ---")
        initial_state = {
            "clinical_note": note,
            "planner_strategy": None,
            "extracted_triples": [],
            "validation_feedback": None,
            "is_valid": False,
            "iterations": 0
        }
        
        final_state = workflow.invoke(initial_state)
        
        if final_state["extracted_triples"]:
            print(f"Extracted {len(final_state['extracted_triples'])} validated triples.")
            builder.add_triples(final_state["extracted_triples"])
        else:
            print("No triples extracted or validation failed.")

    # Step 5: Save and Visualize
    builder.save_graph()
    visualize_graph(builder.graph)
    print("Pipeline complete! View the results in data/processed/")

if __name__ == "__main__":
    main()
