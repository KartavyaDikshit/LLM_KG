# 🚀 Robust Agentic GraphRAG (Next Level)
This notebook implements a high-recall Knowledge Graph construction pipeline with a **GitHub-Synced Codebase**, **JSON Milestone Safety Net**, and **Neo4j Persistence**.

## 1. Setup Environment
```python
# @title Setup: Engine, Dependencies & Neo4j { vertical-output: true }
import os, json, re, yaml, shutil, subprocess, time, sys
from typing import List, Dict, Any

# 1. Project Directory Setup (NOW FULLY SYNCED WITH GITHUB)
PROJECT_DIR = '/content/LLM_KG'
if os.path.exists(PROJECT_DIR): shutil.rmtree(PROJECT_DIR)
!git clone https://github.com/KartavyaDikshit/LLM_KG.git
%cd {PROJECT_DIR}
sys.path.append(PROJECT_DIR)

# 2. Neo4j Credentials
NEO4J_URI = "neo4j+s://37432bb6.databases.neo4j.io"
NEO4J_USER = "37432bb6"
NEO4J_PASSWORD = "mhsAd-D0AxO3eCWnoL5g1cH6jFEjDIxkkntomwHgWU8"

# 3. Install Dependencies
print("📦 Installing Robust Stack...")
!pip install langchain langchain-ollama langchain-community langgraph --quiet
!pip install langchain-google-genai langchain-groq --quiet
!pip install datasets pandas networkx pyvis requests pydantic tqdm tabulate PyYAML seaborn matplotlib neo4j --quiet

# 4. Setup Ollama
print("📥 Starting Ollama...")
!sudo apt-get update && sudo apt-get install -y zstd
!curl -fsSL https://ollama.com/install.sh | sh
subprocess.Popen(["ollama", "serve"])
time.sleep(15)
!ollama pull llama3
print("✅ Environment Ready!")
```

## 2. Robust Extraction Loop
```python
# @title Execution: Multi-Agent Extraction & Neo4j Sync { vertical-output: true }
from src.agents.graph import create_agentic_workflow
from src.graph.milestone import MilestoneManager
from src.graph.neo4j_manager import Neo4jManager
from langchain_ollama import ChatOllama
from datasets import load_dataset

# Initialize
ms = MilestoneManager()
neo = Neo4jManager(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
workflow = create_agentic_workflow()
model_name = "llama3"
domain = "medical"
llm = ChatOllama(model=model_name, temperature=0)

# Load Data (FIXED: Using explicit repo path to avoid HfUriError)
print("⚖️ Loading Dataset...")
legal_ds = load_dataset("fieri/billsum", split="train") # Updated path
corpus = legal_ds['text'][:10]

# 1. RUN EXTRACTION
processed = ms.get_processed_indices() # Milestone check
print(f"🔍 Found {len(processed)} existing milestones. Resuming...")

for i, text in enumerate(corpus):
    if i in processed: continue 
    
    try:
        res = workflow.invoke(
            {"input_text": text[:2000], "domain": domain, "is_valid": False, "iterations": 0},
            config={"configurable": {"llm": llm}}
        )
        ms.save(i, res.get("extracted_triples", []), metadata={"model": model_name, "domain": domain})
    except Exception as e:
        print(f"⚠️ Error on doc {i}: {e}")

# 2. SYNC TO NEO4J
print("\n🔗 Syncing milestones to Neo4j...")
all_triples = ms.get_all_triples()
if all_triples:
    neo.upload_triples(all_triples)
else:
    print("❌ No triples to sync.")
```

## 3. GraphRAG: Query the Knowledge Graph
```python
# @title Query: Ask Natural Language Questions { vertical-output: true }
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain

# Connect LangChain to Neo4j
graph = Neo4jGraph(url=NEO4J_URI, username=NEO4J_USER, password=NEO4J_PASSWORD)
chain = GraphCypherQAChain.from_llm(llm, graph=graph, verbose=True)

# Test Question
q = "Which entities have the most relationships and what are they?"
print(f"\n❓ Question: {q}")
try:
    response = chain.invoke(q)
    print(f"\n💡 Answer: {response['result']}")
except Exception as e:
    print(f"⚠️ Query Error: {e}")
```

## 4. Visualizing Persistent Data
```python
# @title Visualization: Interactive View from Milestones { vertical-output: true }
import networkx as nx
from src.graph.visualizer import visualize_graph
from IPython.display import HTML, display
import base64

# Build NetworkX from the persistent Milestones for error-free rendering
all_triples = ms.get_all_triples()

if not all_triples:
    print("❌ No triples found. Run the extraction cell first.")
else:
    G = nx.MultiDiGraph()
    for t in all_triples:
        G.add_edge(t.subject, t.obj, label=t.predicate)

    output_html = "data/processed/final_robust_kg.html"
    visualize_graph(G, domain="medical", output_path=output_html)

    # Render
    with open(output_html, 'r') as f: html = f.read()
    b64 = base64.b64encode(html.encode()).decode()
    display(HTML(f'<h3>🔍 Final Persistent Knowledge Graph</h3><iframe src="data:text/html;base64,{b64}" width="100%" height="600px" style="border:none;"></iframe>'))
```
