# 🚀 Robust Research Execution (Persistence & Error-Free)

This script is the final, hardened version of the Knowledge Graph extraction loop. It resolves the `ValueError` by using double-escaped braces and implements a **Checkpoint System** to ensure that data is never lost during long compute runs.

### 🛠️ Key Improvements
1.  **JSON Checkpointing**: Saves every triple to `data/processed/triples_checkpoint.json` the moment a model finishes a domain.
2.  **Robust Error Handling**: Added `try-except` blocks around model invocations to prevent the entire notebook from crashing if one model or document fails.
3.  **Instant Visualization**: Generates domain-specific `.html` maps *inside* the loop so you can view results as they are completed.

---

### 🚀 Implementation Code (Copy into Colab)

```python
# @title 🚀 ROBUST RESEARCH EXECUTION (Persistence & Error-Free)
import os, json, re, time, base64
from src.agents.graph import create_agentic_workflow
from src.graph.builder import GraphBuilder
from src.graph.visualizer import visualize_graph
from langchain_ollama import ChatOllama
import pandas as pd
from IPython.display import HTML, display

# --- 1. SETTINGS & PATHS ---
os.makedirs('data/processed', exist_ok=True)
CHECKPOINT_PATH = 'data/processed/triples_checkpoint.json'
workflow = create_agentic_workflow()
test_models = ["llama3", "mistral", "gemma2"]

# Use the full datasets loaded in the previous cells
# (Falling back to samples if not found)
try:
    domains = {
        "medical": medical_notes[:10], 
        "legal": [c[:2000] for c in legal_corpus[:10]]
    }
except NameError:
    print("⚠️ Data variables not found. Using tiny sample for testing...")
    domains = {
        "medical": ["Patient with T2D prescribed Metformin."], 
        "legal": ["This contract is governed by laws of NY."]
    }

# --- 2. THE PERSISTENT BENCHMARK LOOP ---
results_log = []
all_data = {} # Master dictionary to save to disk

# Try to resume from checkpoint if it exists
if os.path.exists(CHECKPOINT_PATH):
    try:
        with open(CHECKPOINT_PATH, 'r') as f:
            all_data = json.load(f)
            print(f"🔄 Found checkpoint. Resuming from model: {list(all_data.keys())[-1]}")
    except: pass

for model_name in test_models:
    if model_name in all_data and len(all_data[model_name]) == len(domains):
        print(f"⏩ Skipping {model_name} (Already in checkpoint)")
        continue

    print(f"\n🧠 Initializing {model_name}...")
    llm = ChatOllama(model=model_name, temperature=0)
    if model_name not in all_data: all_data[model_name] = {}
    
    for domain, texts in domains.items():
        if domain in all_data[model_name]:
            print(f"   ⏩ Skipping {domain}")
            continue

        print(f"🚀 Processing {domain.upper()} with {model_name}...")
        start_time = time.time()
        domain_triples = []
        
        for i, text in enumerate(texts):
            try:
                # Invoke the fixed Agentic workflow
                res = workflow.invoke(
                    {"input_text": text, "domain": domain, "is_valid": False, "iterations": 0},
                    config={"configurable": {"llm": llm}}
                )
                extracted = res.get("extracted_triples", [])
                domain_triples.extend(extracted)
                print(f"   ✅ Doc {i+1}/{len(texts)}: Found {len(extracted)} triples", end='\r')
            except Exception as e:
                print(f"\n   ❌ Error in {model_name} on {domain} doc {i+1}: {e}")

        # --- 3. SAVE TO DISK IMMEDIATELY ---
        elapsed = time.time() - start_time
        # Convert Pydantic Triple objects to dictionaries for JSON serialization
        triples_dicts = []
        for t in domain_triples:
            if hasattr(t, 'dict'):
                triples_dicts.append(t.dict())
            elif isinstance(t, dict):
                triples_dicts.append(t)
            else:
                # Handle unexpected formats
                triples_dicts.append({"subject": str(t.subject), "predicate": str(t.predicate), "obj": str(t.obj), "confidence": 1.0})

        all_data[model_name][domain] = {"triples": triples_dicts, "speed": elapsed}
        
        with open(CHECKPOINT_PATH, 'w') as f:
            json.dump(all_data, f)
            
        # --- 4. GENERATE VISUAL IMMEDIATELY ---
        builder = GraphBuilder()
        from src.agents.state import Triple
        builder.add_triples([Triple(**t) for t in triples_dicts])
        vis_path = f"data/processed/kg_{domain}_{model_name}.html"
        visualize_graph(builder.graph, domain=domain, output_path=vis_path)
        
        # Log stats for the table
        density = len(domain_triples) / len(texts)
        results_log.append({
            "Model": model_name, "Domain": domain, 
            "Total Triples": len(domain_triples), "Triples/Doc": round(density, 2),
            "Speed (sec)": round(elapsed, 2)
        })

# --- 5. FINAL OUTPUT & METRICS ---
if results_log:
    df_results = pd.DataFrame(results_log)
    print("\n\n📊 FINAL PERFORMANCE METRICS")
    display(df_results)

def render_kg(path, title):
    if not os.path.exists(path): return
    with open(path, 'r') as f: html = f.read()
    b64 = base64.b64encode(html.encode()).decode()
    display(HTML(f'<h3>🔍 {title}</h3><iframe src="data:text/html;base64,{b64}" width="100%" height="600px" style="border:none;"></iframe>'))

# Show the results for the primary models
render_kg("data/processed/kg_medical_llama3.html", "Medical Knowledge Graph (Llama 3)")
render_kg("data/processed/kg_legal_mistral.html", "Legal Knowledge Graph (Mistral)")
```
