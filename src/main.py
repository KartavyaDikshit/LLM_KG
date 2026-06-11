import os
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.ingestion.fetcher import fetch_clinvec_data
from src.ingestion.loader import load_clinical_notes
from src.agents.graph import create_agentic_workflow, create_query_workflow
from src.graph.builder import GraphBuilder
from src.graph.visualizer import visualize_graph, visualize_from_neo4j
from src.graph.neo4j_manager import Neo4jManager

def main():
    parser = argparse.ArgumentParser(description="Personalized Medicine Knowledge Graph Generator")
    parser.add_argument("--notes", type=str, help="Path to MIMIC-IV clinical notes CSV", default="data/raw/notes_sample.csv")
    parser.add_argument("--fetch-ontology", action="store_true", help="Download ClinVec ontology from Dataverse")
    parser.add_argument("--limit", type=int, help="Limit the number of notes to process", default=5)
    parser.add_argument("--domain", type=str, help="Domain config to use (medical/legal)", default="medical")
    parser.add_argument("--neo4j", action="store_true", help="Upload results to Neo4j")
    parser.add_argument("--clear-neo4j", action="store_true", help="Clear Neo4j before upload")
    parser.add_argument("--query", type=str, help="Run a natural language query against the graph")
    
    args = parser.parse_args()

    # Step 1: Fetch Ontology if requested (Medical only for now)
    if args.fetch_ontology and args.domain == "medical":
        fetch_clinvec_data()

    # Step 2: Load Clinical Notes
    if not os.path.exists(args.notes):
        print(f"Notes file not found: {args.notes}. Please provide a valid CSV or use --fetch-ontology and a sample.")
        # If we have a query, we might want to proceed if Neo4j is already populated
        if not args.query:
            return

    if os.path.exists(args.notes):
        notes = load_clinical_notes(args.notes)
        notes = notes[:args.limit]
        print(f"Processing {len(notes)} clinical notes...")

        # Step 3: Initialize Pipeline and Graph
        workflow = create_agentic_workflow()
        builder = GraphBuilder()

        all_triples = []

        # Step 4: Run Agentic Pipeline for each note
        for i, note in enumerate(notes):
            print(f"--- Processing {args.domain.capitalize()} Item {i+1}/{len(notes)} ---")
            initial_state = {
                "input_text": note,
                "domain": args.domain,
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
                all_triples.extend(final_state["extracted_triples"])
            else:
                print("No triples extracted or validation failed.")

        # Step 5: Save and Visualize (NetworkX)
        builder.save_graph()
        visualize_graph(builder.graph, domain=args.domain)
        
        # Step 6: Neo4j Upload
        if args.neo4j:
            manager = Neo4jManager()
            if args.clear_neo4j:
                manager.clear_database()
            manager.upload_triples(all_triples, domain=args.domain)
            visualize_from_neo4j(manager, domain=args.domain)
            manager.close()
    
    # Step 7: Querying
    if args.query:
        query_workflow = create_query_workflow()
        print(f"\n--- Querying Graph: {args.query} ---")
        query_state = {
            "query": args.query,
            "answer": None
        }
        result = query_workflow.invoke(query_state)
        print(f"Answer: {result['answer']}")

    # Step 8: Evaluation
    if os.path.exists("data/processed/knowledge_graph.json"):
        from src.evaluation.metrics import evaluate_graph_quality, print_evaluation_report
        from src.ingestion.loader import load_ontology
        
        builder = GraphBuilder()
        builder.load_graph()
        nodes_df, _ = load_ontology()
        stats = evaluate_graph_quality(builder.graph, nodes_df)
        print_evaluation_report(stats)
    
    print("Pipeline complete! View the results in data/processed/")

if __name__ == "__main__":
    main()
