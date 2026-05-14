import os
import argparse
import pandas as pd
from dotenv import load_dotenv
from tabulate import tabulate

# Load environment variables
load_dotenv()

from src.ingestion.loader import load_clinical_notes, load_ontology
from src.agents.graph import create_agentic_workflow
from src.agents.nodes import get_llm
from src.graph.builder import GraphBuilder

def run_benchmark(notes, model_configs):
    results = []
    workflow = create_agentic_workflow()

    for config in model_configs:
        model_type = config['type']
        model_name = config['name']
        print(f"\n🚀 Benchmarking Model: {model_type} ({model_name})")
        
        try:
            llm = get_llm(model_type, model_name)
        except Exception as e:
            print(f"❌ Failed to initialize {model_name}: {e}")
            continue

        total_nodes = 0
        total_edges = 0
        total_triples = 0
        
        for i, note in enumerate(notes):
            print(f"  Processing Note {i+1}/{len(notes)}...")
            initial_state = {
                "clinical_note": note,
                "planner_strategy": None,
                "extracted_triples": [],
                "validation_feedback": None,
                "is_valid": False,
                "iterations": 0
            }
            
            # Pass the LLM instance via config
            final_state = workflow.invoke(
                initial_state, 
                config={"configurable": {"llm": llm}}
            )
            
            triples = final_state["extracted_triples"]
            total_triples += len(triples)
            
            # Temporary builder to count nodes/edges for this model
            temp_builder = GraphBuilder()
            temp_builder.add_triples(triples)
            total_nodes += temp_builder.graph.number_of_nodes()
            total_edges += temp_builder.graph.number_of_edges()

        results.append({
            "Model": f"{model_type}:{model_name}",
            "Avg Triples": total_triples / len(notes),
            "Total Nodes": total_nodes,
            "Total Edges": total_edges,
            "Density": (total_edges / (total_nodes * (total_nodes - 1))) if total_nodes > 1 else 0
        })

    return results

def main():
    parser = argparse.ArgumentParser(description="Medical KG LLM Benchmarking Suite")
    parser.add_argument("--notes", type=str, default="data/raw/notes_sample.csv")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    # Define models to compare (Free Tier APIs)
    model_configs = [
        {"type": "gemini", "name": "gemini-1.5-flash"},
        {"type": "groq", "name": "llama-3.1-70b-versatile"},
        {"type": "groq", "name": "mixtral-8x7b-32768"}
    ]

    print("--- Medical KG LLM Benchmark ---")
    try:
        notes = load_clinical_notes(args.notes)
        notes = notes[:args.limit]
    except Exception as e:
        print(f"Error loading notes: {e}")
        return

    results = run_benchmark(notes, model_configs)

    print("\n" + "="*50)
    print("FINAL BENCHMARK RESULTS")
    print("="*50)
    print(tabulate(results, headers="keys", tablefmt="grid"))
    print("\nAssessment: Higher 'Avg Triples' and 'Total Edges' indicate a more capable model for KG construction.")

if __name__ == "__main__":
    main()
