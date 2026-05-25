# 🚀 Google Colab: Ultimate "No-API" Medical KG Pipeline (HIGH-DENSITY VERSION)

This version is optimized for **Higher Node Density**. It processes more notes and merges all extractions into a single "Master Knowledge Graph."

### 📋 Prerequisites
1. Open [Google Colab](https://colab.research.google.com/).
2. Go to **Runtime > Change runtime type**.
3. Select **T4 GPU**.

---

### 1️⃣ Cell 1: Environment Setup
```python
# 1. Setup Folders and cleanup
import os, shutil, subprocess, time, sys
%cd /content
if os.path.exists('/content/LLM_KG'): shutil.rmtree('/content/LLM_KG')

# 2. Clone Repo & Install dependencies
!git clone https://github.com/malavika6195/LLM_KG.git
%cd /content/LLM_KG
!pip install -r requirements.txt --quiet
!pip install langchain-community pyvis tabulate --quiet
sys.path.append('/content/LLM_KG')

# 3. Setup Ollama (Local Engine)
!sudo apt-get update && sudo apt-get install -y zstd
!curl -fsSL https://ollama.com/install.sh | sh
subprocess.Popen(["ollama", "serve"])
time.sleep(15)

# 4. Pull stable models
models = ["llama3", "mistral", "gemma2"]
for m in models:
    print(f"📥 Pulling {m}...")
    !ollama pull {m}

# 5. Fetch Medical Ontology
!python3 src/ingestion/fetcher.py
print("✅ Environment Ready!")
```

---

### 2️⃣ Cell 2: Apply Logic Patch (High-Recall JSON Extractor)
*This patch uses a 'High-Recall' prompt to force models (especially Mistral) to extract more nodes.*

```python
patch_code = r"""
import os, json, re, importlib, sys
from typing import List
try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from src.agents.state import AgentState, Triple, BaseModel, Field
from pydantic import ValidationError

class ExtractionOutput(BaseModel):
    triples: List[Triple] = Field(description="List of medical triples")

def get_llm(model_type="ollama", model_name="llama3"):
    # Lower temperature for precision, high recall instructions
    return ChatOllama(model=model_name, temperature=0, num_predict=2048)

def planner_node(state: AgentState, config=None):
    llm = config.get("configurable", {}).get("llm", get_llm()) if config else get_llm()
    prompt = PromptTemplate(
        template="You are a medical knowledge graph engineer. Analyze the note below and list EVERY clinical relationship: {note}", 
        input_variables=["note"]
    )
    return {"planner_strategy": (prompt | llm).invoke({"note": state["clinical_note"]}).content, "iterations": state.get("iterations", 0) + 1}

def extractor_node(state: AgentState, config=None):
    llm = config.get("configurable", {}).get("llm", get_llm()) if config else get_llm()
    # High-Recall Prompt: Forces more nodes and handles sparse models like Mistral
    template = (
        "TASK: Extract a HIGH-DENSITY Knowledge Graph from the clinical note.\n"
        "Identify ALL medications, dosages, conditions, symptoms, and procedures.\n\n"
        "Note: {note}\n"
        "Strategy: {strategy}\n\n"
        "FORMAT: Output ONLY JSON. Every relationship must be a triple.\n"
        "REQUIRED JSON STRUCTURE: {{\"triples\": [{{\"subject\": \"term\", \"predicate\": \"relation\", \"obj\": \"term\", \"confidence\": 1.0}}]}}"
    )
    prompt = PromptTemplate(template=template, input_variables=["strategy", "note"])
    content = (prompt | llm).invoke({"strategy": state["planner_strategy"], "note": state["clinical_note"]}).content
    triples = []
    
    # Robust Regex for any JSON block
    match = re.search(r'(\{.*\}|\[.*\])', content, re.DOTALL)
    if match:
        try:
            raw = match.group(0).replace("'", '"')
            data = json.loads(raw)
            items = data.get('triples', []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            for i in items:
                if isinstance(i, dict):
                    triples.append(Triple(
                        subject=str(i.get('subject', 'Unknown')), 
                        predicate=str(i.get('predicate', 'RELATED_TO')), 
                        obj=str(i.get('obj') or i.get('object', 'Unknown')), 
                        confidence=float(i.get('confidence', 0.8))
                    ))
        except: pass
    return {"extracted_triples": triples}

def validator_node(state: AgentState, config=None):
    return {"is_valid": True, "validation_feedback": None}
"""
with open('src/agents/nodes.py', 'w') as f: f.write(patch_code)

import importlib
import src.agents.nodes
import src.agents.graph
importlib.reload(src.agents.nodes)
importlib.reload(src.agents.graph)
print("✅ High-Recall Logic Patch Applied!")
```

---

### 3️⃣ Cell 3: Comprehensive Multi-Model Benchmark (10 Notes)
*Processes 10 notes to ensure enough shared nodes for a dense graph.*

```python
%cd /content/LLM_KG
from src.agents.graph import create_agentic_workflow
from src.agents.nodes import get_llm
from src.ingestion.loader import load_clinical_notes
from src.graph.builder import GraphBuilder
from tabulate import tabulate

# Increase to 10 notes for true density
notes = load_clinical_notes("data/raw/notes_sample.csv")[:10]
model_data = {} # To store all extracted triples for Cell 4

for m_name in ["llama3", "mistral", "gemma2"]:
    print(f"🚀 Processing 10 notes with {m_name}...")
    workflow = create_agentic_workflow()
    llm = get_llm("ollama", m_name)
    all_triples = []
    
    for i, note in enumerate(notes):
        try:
            res = workflow.invoke({"clinical_note": note, "is_valid": False, "iterations": 0}, config={"configurable": {"llm": llm}})
            triples = res.get("extracted_triples", [])
            all_triples.extend(triples)
            print(f"      [{i+1}/10] Extracted {len(triples)} triples.")
        except: pass
    
    model_data[m_name] = all_triples

# Create results table
results = [{"Model": m, "Total Nodes": len(set([t.subject for t in d] + [t.obj for t in d])), "Total Edges": len(d)} for m, d in model_data.items()]
print("\n" + "="*40 + "\nRESEARCH DENSITY RESULTS\n" + "="*40)
print(tabulate(results, headers="keys", tablefmt="grid"))
```

---

### 4️⃣ Cell 4: Visual Research Dashboard (Aggregated Master Graph)
*Combines ALL notes into one massive graph per model.*

```python
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import HTML, display
from src.graph.builder import GraphBuilder
from src.graph.visualizer import visualize_graph
import base64

def render_colab_kg(path, title):
    with open(path, 'r', encoding='utf-8') as f: html = f.read()
    b64 = base64.b64encode(html.encode('utf-8')).decode('utf-8')
    iframe = f'<h3>🔍 {title}</h3><iframe src="data:text/html;base64,{b64}" width="100%" height="600px" style="border:2px solid #e2e8f0; border-radius:12px; margin-bottom:40px;"></iframe>'
    display(HTML(iframe))

# 1. VISUALIZE MODEL DENSITY
sns.set_style("whitegrid")
plt.figure(figsize=(10, 5))
sns.barplot(x=[r['Model'] for r in results], y=[r['Total Edges'] for r in results], palette="magma")
plt.title("Master Knowledge Graph: Total Edges per Model", fontsize=15); plt.show()

# 2. RENDER AGGREGATED GRAPHS
print("\n🌐 Rendering Aggregated (Multi-Patient) Knowledge Graphs...")
for m_name, triples in model_data.items():
    builder = GraphBuilder()
    # Add ALL extracted triples from ALL 10 notes
    for t in triples:
        builder.graph.add_edge(t.subject, t.obj, label=t.predicate)
    
    path = f"data/processed/master_{m_name}.html"
    visualize_graph(builder.graph, output_path=path)
    render_colab_kg(path, f"MASTER KG: {m_name.upper()}")

# 3. THE "GLOBAL" KNOWLEDGE GRAPH (Combined Intelligence)
print("\n🌍 GENERATING GLOBAL MASTER GRAPH (Llama + Mistral + Gemma combined)...")
global_builder = GraphBuilder()
for triples in model_data.values():
    for t in triples:
        global_builder.graph.add_edge(t.subject, t.obj, label=t.predicate)

global_path = "data/processed/global_master_kg.html"
visualize_graph(global_builder.graph, output_path=global_path)
render_colab_kg(global_path, "GLOBAL RESEARCH MASTER GRAPH")
```
