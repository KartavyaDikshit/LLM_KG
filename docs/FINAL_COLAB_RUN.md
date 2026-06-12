# 🚀 Robust Agentic GraphRAG (Final "Zero-Error" Edition)
This notebook implements a high-recall Knowledge Graph construction pipeline with **Persistent Recovery** and **Multi-Database Support**.

## 1. Setup Environment
```python
# @title Setup: Robust Environment Init { vertical-output: true }
import os, json, re, yaml, shutil, subprocess, time, sys

# 1. Reset Directory Context (Fixes shell-init/getcwd errors)
os.chdir('/content')
PROJECT_DIR = '/content/LLM_KG'

# 2. Robust Cleanup & Fresh Clone
if os.path.exists(PROJECT_DIR):
    print("🧹 Cleaning up old project files...")
    shutil.rmtree(PROJECT_DIR)

print("📥 Cloning the latest 'Next Level' codebase from GitHub...")
!git clone https://github.com/KartavyaDikshit/LLM_KG.git

# 3. Path Verification & Module Reload (CRITICAL for Colab re-runs)
if not os.path.exists(f"{PROJECT_DIR}/src"):
    raise Exception("❌ ERROR: Git clone failed. Please check your internet connection and repository URL.")

%cd {PROJECT_DIR}
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# Force reload modules to pick up changes after clone
import importlib
for module_name in list(sys.modules.keys()):
    if module_name.startswith('src.'):
        del sys.modules[module_name]

# 4. Install Dependencies
print("📦 Installing required libraries...")
!pip install langchain langchain-ollama langchain-community langgraph langchain-neo4j --quiet
!pip install datasets pandas networkx pyvis requests pydantic tqdm tabulate PyYAML seaborn matplotlib neo4j --quiet

# 5. Neo4j Credentials
NEO4J_URI = "neo4j+s://37432bb6.databases.neo4j.io"
NEO4J_USER = "37432bb6"
NEO4J_PASSWORD = "mhsAd-D0AxO3eCWnoL5g1cH6jFEjDIxkkntomwHgWU8"

# 6. Setup Ollama (Local LLM Engine)
print("📥 Starting Ollama Server...")
os.system("pkill -9 ollama || true")
time.sleep(2)
!sudo apt-get update && sudo apt-get install -y zstd --quiet
!curl -fsSL https://ollama.com/install.sh | sh
subprocess.Popen(["ollama", "serve"], cwd="/", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(10) # Wait for server to start

print("\n✅ Setup complete! Proceed to Benchmark.")
```

## 2. Multi-Model Benchmark (Legal & Medical)
```python
# @title Execution: Multi-Model Benchmark & Analytics { vertical-output: true }
import os, sys, time, pandas as pd
from tabulate import tabulate
from src.agents.graph import create_agentic_workflow
from src.graph.milestone import MilestoneManager
from src.graph.neo4j_manager import Neo4jManager
from langchain_ollama import ChatOllama
from datasets import load_dataset
import networkx as nx

# --- CONFIGURATION ---
TEST_MODE = True # @param {type:"boolean"}
# If TEST_MODE is True, we only process 1 document per domain to save time.
# Set to False to run the full benchmark.
# ---------------------

# Initialize
ms = MilestoneManager()
neo = Neo4jManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
workflow = create_agentic_workflow()
benchmark_results = []

# 1. LOAD DATASETS
print("⚖️ Loading Datasets...")
limit = 1 if TEST_MODE else 10

try:
    legal_ds = load_dataset("FiscalNote/billsum", split="train")
    legal_corpus = legal_ds['text'][:limit]
except:
    legal_corpus = ["Company A agrees to pay Company B $5000."] * limit

medical_corpus = ["Patient diagnosed with Type 2 Diabetes, prescribed Metformin."] * limit

# 2. RUN BENCHMARK
test_models = ["llama3", "mistral", "gemma2"]
domains = [("legal", legal_corpus), ("medical", medical_corpus)]

for model_name in test_models:
    print(f"\n🚀 Checking Model: {model_name.upper()}")
    # Ensure model is pulled (instant if already exists)
    !ollama pull {model_name}
    
    llm = ChatOllama(model=model_name, temperature=0)
    
    for domain, corpus in domains:
        print(f"  - Domain: {domain}")
        start_time = time.time()
        domain_triples = 0
        
        for i, text in enumerate(corpus):
            try:
                res = workflow.invoke(
                    {"input_text": text[:1500], "domain": domain, "is_valid": False, "iterations": 0},
                    config={"configurable": {"llm": llm}}
                )
                triples = res.get("extracted_triples", [])
                domain_triples += len(triples)
                # Save only Llama3 results to keep graph clean
                if model_name == "llama3":
                    ms.save(f"{model_name}_{domain}_{i}", triples)
            except Exception as e:
                print(f"    ⚠️ Doc {i} failed: {e}")
        
        elapsed = time.time() - start_time
        benchmark_results.append({
            "Model": model_name, "Domain": domain, "Triples": domain_triples,
            "Avg Time/Doc": f"{elapsed/len(corpus):.1f}s",
            "Efficiency": f"{domain_triples/elapsed:.2f} T/s"
        })

# 3. DISPLAY METRICS
print("\n📊 FINAL PERFORMANCE METRICS")
df = pd.DataFrame(benchmark_results)
print(tabulate(df, headers='keys', tablefmt='psql'))
```

## 3. Persistent Graph Sync
```python
# @title Sync: Upload High-Recall Data to Neo4j { vertical-output: true }
print("🔗 Syncing Llama3 milestones to Neo4j persistence layer...")
all_triples = ms.get_all_triples()
if all_triples:
    neo.upload_triples(all_triples)
    print(f"✅ Successfully synced {len(all_triples)} triples.")
else:
    print("❌ No data to sync.")
```

## 4. Visualizing the Global Knowledge Graph
```python
# @title Visualization: Multi-Domain Interactive View { vertical-output: true }
from src.graph.visualizer import visualize_graph
import networkx as nx
from IPython.display import HTML, display
import base64

# Combine all domain data for the visual
all_triples = ms.get_all_triples()
G = nx.MultiDiGraph()

for t in all_triples:
    sub = t.subject if hasattr(t, 'subject') else t['subject']
    obj = t.obj if hasattr(t, 'obj') else t['obj']
    pred = t.predicate if hasattr(t, 'predicate') else t['predicate']
    G.add_edge(sub, obj, label=pred)

output_path = "data/processed/final_interactive_graph.html"
# Using 'medical' theme as default for visual styling
visualize_graph(G, domain="medical", output_path=output_path)

with open(output_path, 'r') as f: html = f.read()
b64 = base64.b64encode(html.encode()).decode()
display(HTML(f'<h3>🔍 Combined Knowledge Graph (Legal + Medical)</h3>'
             f'<iframe src="data:text/html;base64,{b64}" width="100%" height="700px" style="border:none;"></iframe>'))
```
