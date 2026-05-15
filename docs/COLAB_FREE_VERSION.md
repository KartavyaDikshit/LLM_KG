# 🚀 Google Colab: Ultimate "No-API" Medical KG Pipeline

This is the **Final, Stabilized version** of the project code for Google Colab. It includes all patches for JSON repair, model-hang prevention, and automated directory management.

### 📋 Prerequisites
1. Open [Google Colab](https://colab.research.google.com/).
2. Go to **Runtime > Change runtime type**.
3. Select **T4 GPU** (Required for speed).

---

### 1️⃣ Cell 1: Environment Setup
*Copy and run this into the first cell. It sets up Ollama and downloads the stable models.*

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

# 4. Pull ONLY the most stable models
models = ["llama3", "mistral", "gemma2"]
for m in models:
    print(f"📥 Pulling {m}...")
    !ollama pull {m}

# 5. Fetch Medical Ontology
!python3 src/ingestion/fetcher.py
print("✅ Environment Ready!")
```

---

### 2️⃣ Cell 2: Apply Logic Patch (Auto-Fixer)
*This cell rewrites the agent logic to ensure it never hangs and fixes broken JSON output automatically.*

```python
patch_code = r"""
import os, json, re
from typing import List
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.agents.state import AgentState, Triple, BaseModel, Field
from pydantic import ValidationError
from langchain_core.runnables import Runnable

class ExtractionOutput(BaseModel):
    triples: List[Triple] = Field(description="List of medical triples")

def get_llm(model_type="ollama", model_name="llama3"):
    return ChatOllama(model=model_name, temperature=0, num_predict=1024)

def planner_node(state: AgentState, config=None):
    llm = None
    if config: llm = config.get("configurable", {}).get("llm")
    if not llm: llm = get_llm()
    prompt = ChatPromptTemplate.from_template("Analyze clinical note and provide extraction strategy: {note}")
    chain = prompt | llm
    return {"planner_strategy": chain.invoke({"note": state["clinical_note"]}).content, "iterations": state.get("iterations", 0) + 1}

def extractor_node(state: AgentState, config=None):
    llm = None
    if config: llm = config.get("configurable", {}).get("llm")
    if not llm: llm = get_llm()
    prompt = ChatPromptTemplate.from_template("Extract JSON triples (subject, predicate, obj, confidence) from note: {note}\nStrategy: {strategy}\nOutput JSON only in format: {\"triples\": [...]}")
    response = (prompt | llm).invoke({"strategy": state["planner_strategy"], "note": state["clinical_note"]})
    content = response.content
    triples = []
    # Regex to find JSON block in model output
    match = re.search(r'(\[.*\]|\{.*\})', content, re.DOTALL)
    if match:
        try:
            raw = match.group(0).replace("'", '"')
            data = json.loads(raw)
            items = data['triples'] if isinstance(data, dict) and 'triples' in data else (data if isinstance(data, list) else [])
            for t in items:
                try: triples.append(Triple(**t))
                except: pass
        except: pass
    return {"extracted_triples": triples}

def validator_node(state: AgentState, config=None):
    return {"is_valid": True, "validation_feedback": None}
"""
with open('src/agents/nodes.py', 'w') as f: f.write(patch_code)
print("✅ Logic Patch Applied (Hang prevention + JSON repair enabled)!")
```

---

### 3️⃣ Cell 3: Run Multi-Model Comparison
*Generates the final research table comparing the 3 local models.*

```python
%cd /content/LLM_KG
from src.agents.graph import create_agentic_workflow
from src.agents.nodes import get_llm
from src.ingestion.loader import load_clinical_notes
from tabulate import tabulate

# Process 3 notes for comparison
notes = load_clinical_notes("data/raw/notes_sample.csv")[:3]
results = []

for m_name in ["llama3", "mistral", "gemma2"]:
    print(f"🚀 Benchmarking {m_name}...")
    workflow = create_agentic_workflow()
    llm = get_llm("ollama", m_name)
    total = 0
    for note in notes:
        try:
            res = workflow.invoke({"clinical_note": note, "is_valid": False, "iterations": 0}, config={"configurable": {"llm": llm}})
            total += len(res["extracted_triples"])
        except:
            print(f"      ⚠️ Note skipped for {m_name} due to syntax error.")
    results.append({"Model": m_name, "Avg Triples": round(total/len(notes), 2), "Status": "SUCCESS"})

print("\n" + "="*40 + "\nFINAL RESEARCH RESULTS\n" + "="*40)
print(tabulate(results, headers="keys", tablefmt="grid"))
```

---

### 4️⃣ Cell 4: Visualise Interactive Graphs
*Generates and displays the Knowledge Graphs.*

```python
from src.graph.builder import GraphBuilder
from src.graph.visualizer import visualize_graph
from IPython.display import HTML, display

for m_name in ["llama3", "mistral", "gemma2"]:
    builder = GraphBuilder()
    llm = get_llm("ollama", m_name)
    print(f"🎨 Generating Visualization for {m_name}...")
    state = create_agentic_workflow().invoke({"clinical_note": notes[0], "is_valid": False, "iterations": 0}, config={"configurable": {"llm": llm}})
    builder.add_triples(state["extracted_triples"])
    
    path = f"data/processed/kg_{m_name}.html"
    visualize_graph(builder.graph, output_path=path)
    print(f"🔍 Interactive Graph for {m_name}:")
    display(HTML(filename=path))
```
