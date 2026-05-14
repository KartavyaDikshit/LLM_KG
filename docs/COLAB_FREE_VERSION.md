# 🚀 Google Colab: Complete "No-API" Medical KG Pipeline (FIXED)

This guide contains all the code cells you need to run the Personalized Medicine Knowledge Graph project in Google Colab using only **FREE, Open-Source models** (no API keys required).

### 📋 Prerequisites
1. Open [Google Colab](https://colab.research.google.com/).
2. Go to **Runtime > Change runtime type**.
3. Select **T4 GPU** (This is required to run LLMs locally).

---

### 1️⃣ Step 1: Environment Setup
*Copy this into the first cell and run it. This installs the system dependencies and the Ollama engine.*

```python
# 1. Cleanup old clones to avoid nested folders
import os
import shutil
if os.path.exists('/content/LLM_KG'):
    shutil.rmtree('/content/LLM_KG')

# 2. Clone the repository
!git clone https://github.com/malavika6195/LLM_KG.git
%cd /content/LLM_KG

# 3. Install Python dependencies
!pip install -r requirements.txt --quiet
!pip install langchain-community pyvis tabulate --quiet

# 4. CRITICAL: Install zstd and Ollama
!sudo apt-get update && sudo apt-get install -y zstd
!curl -fsSL https://ollama.com/install.sh | sh

# 5. Verify and Start Ollama background service
import subprocess
import time
import sys
sys.path.append('/content/LLM_KG')

ollama_path = shutil.which("ollama")
if ollama_path:
    print(f"✅ Ollama installed successfully at: {ollama_path}")
    subprocess.Popen([ollama_path, "serve"])
    time.sleep(15) # Wait for service to initialize
    print("✅ Environment Ready!")
else:
    print("❌ Ollama installation failed. Please check the output above for errors.")
```

---

### 2️⃣ Step 2: Download Models & Medical Data
*This cell pulls the Top 5 free models and the ClinVec ontology files (~450MB). This may take 5-10 minutes.*

```python
# Pull the Top 5 free medical-capable models
models = ["llama3", "mistral", "gemma2", "phi3:medium", "biomistral"]

for model in models:
    print(f"📥 Downloading {model}...")
    !ollama pull {model}

# Download the ClinVec Knowledge Graph Data
!python3 src/ingestion/fetcher.py
print("✅ Models and Data Downloaded!")
```

---

### 3️⃣ Step 3: Run Multi-Model Comparison
*Run this to compare how each model performs at building a Knowledge Graph.*

```python
from src.agents.graph import create_agentic_workflow
from src.agents.nodes import get_llm
from src.graph.builder import GraphBuilder
from src.ingestion.loader import load_clinical_notes
from tabulate import tabulate

# Load sample notes
notes = load_clinical_notes("data/raw/notes_sample.csv")[:3]
comparison_results = []

print("🚀 Starting Benchmark...")
for model_name in ["llama3", "mistral", "gemma2", "phi3:medium", "biomistral"]:
    print(f"  > Processing with {model_name}...")
    workflow = create_agentic_workflow()
    llm = get_llm("ollama", model_name)
    
    total_triples = 0
    for note in notes:
        state = {"clinical_note": note, "is_valid": False, "iterations": 0}
        final_state = workflow.invoke(state, config={"configurable": {"llm": llm}})
        total_triples += len(final_state["extracted_triples"])
    
    comparison_results.append({
        "Model": model_name,
        "Total Triples": total_triples,
        "Avg Per Note": total_triples / len(notes),
        "API Cost": "FREE"
    })

# Output the final table
print("\n" + "="*50)
print("RESEARCH RESULTS: NO-API MODEL COMPARISON")
print("="*50)
print(tabulate(comparison_results, headers="keys", tablefmt="grid"))
```

---

### 4️⃣ Step 4: Visualise the Knowledge Graphs
*Render interactive maps for every model to visually inspect the data.*

```python
from src.graph.visualizer import visualize_graph
from IPython.display import HTML, display

for model_name in ["llama3", "mistral", "gemma2", "phi3_medium", "biomistral"]:
    builder = GraphBuilder()
    llm = get_llm("ollama", model_name.replace('_', ':'))
    workflow = create_agentic_workflow()
    
    # Generate KG for the first note
    state = workflow.invoke({"clinical_note": notes[0]}, config={"configurable": {"llm": llm}})
    builder.add_triples(state["extracted_triples"])
    
    # Save visualization
    safe_name = model_name.replace(':', '_')
    output_path = f"data/processed/kg_{safe_name}.html"
    visualize_graph(builder.graph, output_path=output_path)
    
    print(f"\n🔍 Interactive KG: {model_name}")
    display(HTML(filename=output_path))
```
