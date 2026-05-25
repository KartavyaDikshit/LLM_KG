import os
import yaml
from tabulate import tabulate
from src.agents.graph import create_agentic_workflow
from src.agents.nodes import get_llm
from src.graph.builder import GraphBuilder
from src.graph.visualizer import visualize_graph

def run_benchmark(domain="medical", model_name="llama3", input_text=None):
    """Run an end-to-end KG extraction for a specific domain."""
    print(f"\n🚀 Running {domain.upper()} Benchmark with {model_name}...")
    
    # Load domain config for visibility
    with open(f"src/config/domains/{domain}.yaml", 'r') as f:
        cfg = yaml.safe_load(f)
    
    workflow = create_agentic_workflow()
    llm = get_llm("ollama", model_name)
    
    # Run pipeline
    state = workflow.invoke({
        "input_text": input_text,
        "domain": domain,
        "is_valid": False,
        "iterations": 0
    }, config={"configurable": {"llm": llm}})
    
    triples = state.get("extracted_triples", [])
    print(f"✅ Extracted {len(triples)} triples.")
    
    # Build Graph
    builder = GraphBuilder()
    builder.add_triples(triples)
    
    # Save & Visualize
    vis_path = f"data/processed/benchmark_{domain}_{model_name}.html"
    visualize_graph(builder.graph, domain=domain, output_path=vis_path)
    print(f"📊 Visualization saved to {vis_path}")
    
    return triples

if __name__ == "__main__":
    # Test Medical
    med_text = "Patient with chronic hypertension prescribed Lisinopril 20mg. Reports frequent headaches."
    run_benchmark(domain="medical", input_text=med_text)
    
    # Test Legal
    legal_text = "This Agreement is governed by the laws of the State of Delaware. ACME Corp shall pay $1000 to Beta Inc by June 1st."
    run_benchmark(domain="legal", input_text=legal_text)
