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

# 3. Path Verification
if not os.path.exists(f"{PROJECT_DIR}/src"):
    raise Exception("❌ ERROR: Git clone failed. Please check your internet connection and repository URL.")

%cd {PROJECT_DIR}
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# 4. Install Dependencies
print("📦 Installing required libraries...")
!pip install langchain langchain-ollama langchain-community langgraph --quiet
!pip install langchain-google-genai langchain-groq --quiet
!pip install datasets pandas networkx pyvis requests pydantic tqdm tabulate PyYAML seaborn matplotlib neo4j --quiet

# 5. Neo4j Credentials
NEO4J_URI = "neo4j+s://37432bb6.databases.neo4j.io"
NEO4J_USER = "37432bb6"
NEO4J_PASSWORD = "mhsAd-D0AxO3eCWnoL5g1cH6jFEjDIxkkntomwHgWU8"

# 6. Setup Ollama (Local LLM Engine)
print("📥 Starting Ollama...")
!sudo apt-get update && sudo apt-get install -y zstd
!curl -fsSL https://ollama.com/install.sh | sh
subprocess.Popen(["ollama", "serve"])
time.sleep(15) # Wait for server to start
!ollama pull llama3

print("\n✅ Environment is perfectly set up and synchronized with GitHub!")
```

## 2. Robust Extraction Loop
```python
# @title Execution: Safe Multi-Agent Extraction & Neo4j Sync { vertical-output: true }
from src.agents.graph import create_agentic_workflow
from src.graph.milestone import MilestoneManager
from src.graph.neo4j_manager import Neo4jManager
from langchain_ollama import ChatOllama
from datasets import load_dataset
import time

# Initialize
ms = MilestoneManager()
neo = Neo4jManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
workflow = create_agentic_workflow()
model_name = "llama3"
domain = "medical"
llm = ChatOllama(model=model_name, temperature=0)

# Load Data (Stable Repository Path)
print("⚖️ Loading Legal Dataset (BillSum)...")
try:
    legal_ds = load_dataset("fieri/billsum", split="train")
    corpus = legal_ds['text'][:20] # Scale to 20 documents
except Exception as e:
    print(f"⚠️ Dataset load error: {e}. Using fallback corpus.")
    corpus = ["Patient has history of Stage IV Melanoma. Treatment includes Pembrolizumab."] * 10

# 1. RUN EXTRACTION WITH MILESTONES
processed = ms.get_processed_indices()
print(f"🔍 Progress: {len(processed)} documents already processed. Skipping to next...")

for i, text in enumerate(corpus):
    if i in processed: continue 
    
    start_time = time.time()
    try:
        # Process first 2000 chars to avoid model context limits
        res = workflow.invoke(
            {"input_text": text[:2000], "domain": domain, "is_valid": False, "iterations": 0},
            config={"configurable": {"llm": llm}}
        )
        triples = res.get("extracted_triples", [])
        ms.save(i, triples, metadata={"model": model_name, "domain": domain})
        
        elapsed = time.time() - start_time
        print(f"✅ Doc {i} completed in {elapsed:.2f}s | {len(triples)} triples extracted.")
    except Exception as e:
        print(f"⚠️ Error processing doc {i}: {e}. Skipping.")

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
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain

# Connect and Query
try:
    graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)
    chain = GraphCypherQAChain.from_llm(llm, graph=graph, verbose=True)

    # Example question based on extracted data
    query = "List the entities mentioned and describe their relationships."
    print(f"❓ Question: {query}")
    response = chain.invoke(query)
    print(f"\n💡 Answer: {response['result']}")
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
        G.add_edge(t.subject, t.obj, label=t.predicate)

    output_path = "data/processed/final_interactive_graph.html"
    visualize_graph(G, domain="medical", output_path=output_path)

    # Display in Notebook
    with open(output_path, 'r') as f: html = f.read()
    b64 = base64.b64encode(html.encode()).decode()
    display(HTML(f'<h3>🔍 Interactive View (Recovered from Milestone Storage)</h3>'
                 f'<iframe src="data:text/html;base64,{b64}" width="100%" height="650px" style="border:none;"></iframe>'))
```
