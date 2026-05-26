import os
import json
import yaml
import time
from typing import List, Dict, Any
from tabulate import tabulate
from src.agents.graph import create_agentic_workflow
from src.graph.builder import GraphBuilder
from src.graph.visualizer import visualize_graph
from langchain_ollama import ChatOllama

def run_research_benchmark(test_models: List[str], domains_data: Dict[str, List[str]]):
    """
    Run a robust, multi-model benchmark with persistence.
    Saves triples to disk as they are extracted to prevent data loss.
    """
    os.makedirs('data/processed', exist_ok=True)
    results_backup_path = 'data/processed/triples_checkpoint.json'
    
    # Load existing progress if any
    all_extracted_data = {}
    if os.path.exists(results_backup_path):
        with open(results_backup_path, 'r') as f:
            all_extracted_data = json.load(f)
            print(f"🔄 Resuming from checkpoint: {len(all_extracted_data)} models found.")

    results_log = []
    workflow = create_agentic_workflow()

    for model_name in test_models:
        print(f"\n🧠 Initializing Model: {model_name}")
        llm = ChatOllama(model=model_name, temperature=0)
        
        if model_name not in all_extracted_data:
            all_extracted_data[model_name] = {}

        for domain, texts in domains_data.items():
            print(f"🚀 Benchmarking Domain: {domain}")
            
            # Use cached results if available
            if domain in all_extracted_data[model_name]:
                print(f"   ⏩ Skipping {domain} (already in checkpoint).")
                triples_dicts = all_extracted_data[model_name][domain]["triples"]
                elapsed = all_extracted_data[model_name][domain]["speed"]
            else:
                start_time = time.time()
                domain_triples = []
                
                for i, text in enumerate(texts):
                    print(f"   📄 Processing doc {i+1}/{len(texts)}...", end='\r')
                    try:
                        res = workflow.invoke(
                            {"input_text": text, "domain": domain, "is_valid": False, "iterations": 0},
                            config={"configurable": {"llm": llm}}
                        )
                        extracted = res.get("extracted_triples", [])
                        domain_triples.extend(extracted)
                    except Exception as e:
                        print(f"\n   ❌ Error in doc {i+1}: {e}")
                
                elapsed = time.time() - start_time
                # Convert Triple objects to dicts for JSON storage
                triples_dicts = [t.dict() if hasattr(t, 'dict') else t for t in domain_triples]
                
                # Update checkpoint
                all_extracted_data[model_name][domain] = {
                    "triples": triples_dicts,
                    "speed": elapsed
                }
                with open(results_backup_path, 'w') as f:
                    json.dump(all_extracted_data, f)
                print(f"\n   ✅ Done. Extracted {len(triples_dicts)} triples in {elapsed:.2f}s.")

            # Calculate density and update log
            density = len(triples_dicts) / len(texts) if texts else 0
            results_log.append({
                "Model": model_name, "Domain": domain, 
                "Total Triples": len(triples_dicts), "Triples/Doc": round(density, 2),
                "Speed (sec)": round(elapsed, 2)
            })

            # Build and Save Graph for THIS specific model/domain
            builder = GraphBuilder()
            # Reconstruct Triple objects from dicts for builder
            from src.agents.state import Triple
            triples_objs = [Triple(**t) for t in triples_dicts]
            builder.add_triples(triples_objs)
            
            vis_path = f"data/processed/kg_{domain}_{model_name}.html"
            visualize_graph(builder.graph, domain=domain, output_path=vis_path)

    # Final Summary
    df_results = pd.DataFrame(results_log)
    print("\n📊 RESEARCH PERFORMANCE METRICS")
    print(tabulate(df_results, headers='keys', tablefmt='grid'))
    
    return all_extracted_data

if __name__ == "__main__":
    import pandas as pd
    # Test sample
    test_models = ["llama3"]
    domains = {"medical": ["Test note 1"], "legal": ["Test legal 1"]}
    run_research_benchmark(test_models, domains)
