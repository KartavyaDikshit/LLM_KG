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
print("📥 Starting Ollama...")
os.system("pkill -9 ollama || true")
time.sleep(2)
!sudo apt-get update && sudo apt-get install -y zstd --quiet
!curl -fsSL https://ollama.com/install.sh | sh
subprocess.Popen(["ollama", "serve"], cwd="/", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(15) # Wait for server to start
!ollama pull llama3

print("\n✅ Environment is perfectly set up and synchronized with GitHub!")
```

## 2. Robust Extraction Loop
```python
# @title Execution: Safe Multi-Agent Extraction & Neo4j Sync { vertical-output: true }
import os, sys, subprocess
# Guard: Ensure we are always in the right folder
PROJECT_DIR = '/content/LLM_KG'
os.chdir(PROJECT_DIR)
if PROJECT_DIR not in sys.path: sys.path.insert(0, PROJECT_DIR)

# 1. Hardware Speed Check
try:
    gpu_check = subprocess.check_output("nvidia-smi", shell=True).decode()
    print("🚀 NVIDIA GPU Detected! Extraction will be fast.")
except:
    print("⚠️ WARNING: No NVIDIA GPU detected. TPU/CPU mode is 10x slower for Ollama.")
    print("💡 Tip: Change runtime to 'T4 GPU' or 'L4 GPU' for better performance.")

from src.agents.graph import create_agentic_workflow
from src.graph.milestone import MilestoneManager
from src.graph.neo4j_manager import Neo4jManager
from langchain_ollama import ChatOllama
from datasets import load_dataset
import time

# Initialize
ms = MilestoneManager()
neo = Neo4jManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
# FAST MODE: Bypasses planner/validator for speed on CPU/TPU
workflow = create_agentic_workflow(fast_mode=True)
model_name = "llama3"
llm = ChatOllama(model=model_name, temperature=0)

# 2. Load Datasets
print("⚖️ Loading Datasets (Medical & Legal)...")

# Legal Dataset
try:
    legal_ds = load_dataset("FiscalNote/billsum", split="train")
    legal_corpus = legal_ds['text'][:10]
    print(f"✅ Loaded {len(legal_corpus)} Legal documents.")
except Exception as e:
    print(f"⚠️ Legal dataset error: {e}. Using fallback.")
    legal_corpus = ["Company A agrees to pay Company B $5000 for consulting services under California law."] * 5

# Medical Dataset
medical_corpus = [
    "Patient diagnosed with Type 2 Diabetes, prescribed Metformin 500mg. Reports neuropathy in extremities.",
    "Acute Respiratory Distress Syndrome secondary to viral pneumonia. Patient on ventilator support.",
    "Subject has history of Stage IV Melanoma. Undergoing immunotherapy with Pembrolizumab."
] * 3

datasets_to_process = [
    ("legal", legal_corpus),
    ("medical", medical_corpus)
]

# 3. RUN EXTRACTION
processed = ms.get_processed_indices()
print(f"🔍 Progress: {len(processed)} existing milestones. Resuming...")

global_i = 0
for domain, corpus in datasets_to_process:
    print(f"\n🚀 Processing {len(corpus)} documents for domain: {domain.upper()}")
    for text in corpus:
        if global_i in processed:
            global_i += 1
            continue 

        try:
            # Reduced to 1000 chars for extreme speed-up
            res = workflow.invoke(
                {"input_text": text[:1000], "domain": domain, "is_valid": False, "iterations": 0},
                config={"configurable": {"llm": llm}}
            )
            triples = res.get("extracted_triples", [])
            ms.save(global_i, triples, metadata={"model": "llama3", "domain": domain})
            print(f"✅ Doc {global_i} ({domain}) processed. Extracted {len(triples)} triples.")
        except Exception as e:
            print(f"⚠️ Doc {global_i} ({domain}) failed: {e}. Moving on.")
        global_i += 1

# 2. SYNC TO NEO4J (Incremental Upload)
print("\n🔗 Syncing all milestones to Neo4j persistence layer...")
all_triples = ms.get_all_triples()
if all_triples:
    neo.upload_triples(all_triples)
else:
    print("❌ No triples available to sync. Please run extraction first.")
```

## 3. GraphRAG: Ask Natural Language Questions
```python
# @title Query: Speak to your Knowledge Graph { vertical-output: true }
from langchain_neo4j import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

# Connect and Query
try:
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD, database=None)
    chain = GraphCypherQAChain.from_llm(llm, graph=graph, verbose=True)

    # Example question based on extracted data
    query = "List the entities mentioned and describe their relationships."
    print(f"❓ Question: {query}")
    response = chain.invoke(query)
    print(f"\n💡 Answer: {response.get('result', response.get('query'))}")
except Exception as e:
    print(f"⚠️ GraphRAG Query Error: {e}")
```

## 4. Visualizing Persistent Data
```python
# @title Visualization: Interactive Knowledge Graph { vertical-output: true }
import networkx as nx
from src.graph.visualizer import visualize_graph
from IPython.display import HTML, display
import base64

# Reconstruct from Milestones for stability
all_triples = ms.get_all_triples()

if not all_triples:
    print("❌ No data to visualize. Complete a processing run first.")
else:
    G = nx.MultiDiGraph()
    for t in all_triples:
        # Robust triple access (dict or object)
        sub = t.subject if hasattr(t, 'subject') else t['subject']
        obj = t.obj if hasattr(t, 'obj') else t['obj']
        pred = t.predicate if hasattr(t, 'predicate') else t['predicate']
        G.add_edge(sub, obj, label=pred)

    output_path = "data/processed/final_interactive_graph.html"
    visualize_graph(G, domain="legal", output_path=output_path)

    # Display in Notebook
    with open(output_path, 'r') as f: html = f.read()
    b64 = base64.b64encode(html.encode()).decode()
    display(HTML(f'<h3>🔍 Interactive View (Recovered from Milestone Storage)</h3>'
                 f'<iframe src="data:text/html;base64,{b64}" width="100%" height="650px" style="border:none;"></iframe>'))
```
