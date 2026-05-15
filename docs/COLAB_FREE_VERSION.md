# 🚀 Google Colab: Complete "No-API" Medical KG Pipeline (STABLE VERSION)

This guide contains all the code cells you need to run the Personalized Medicine Knowledge Graph project in Google Colab using only **FREE, Open-Source models**.

### 📋 Prerequisites
1. Open [Google Colab](https://colab.research.google.com/).
2. Go to **Runtime > Change runtime type**.
3. Select **T4 GPU**.

---

### 1️⃣ Step 1: Environment Setup
*Copy this into the first cell and run it.*

```python
# 1. Cleanup and Reset
import os
import shutil
%cd /content
if os.path.exists('/content/LLM_KG'):
    shutil.rmtree('/content/LLM_KG')

# 2. Clone the repository
!git clone https://github.com/malavika6195/LLM_KG.git
%cd /content/LLM_KG

# 3. Install Python dependencies
!pip install -r requirements.txt --quiet
!pip install langchain-community pyvis tabulate --quiet

# 4. Install zstd and Ollama
!sudo apt-get update && sudo apt-get install -y zstd
!curl -fsSL https://ollama.com/install.sh | sh

# 5. Start Ollama background service
import subprocess
import time
import sys
sys.path.append('/content/LLM_KG')

ollama_path = shutil.which("ollama")
if ollama_path:
    print(f"✅ Ollama installed successfully at: {ollama_path}")
    subprocess.Popen([ollama_path, "serve"])
    time.sleep(20) # Wait for service to initialize
    print("✅ Environment Ready!")
else:
    print("❌ Ollama installation failed.")
```

---

### 2️⃣ Step 2: Download Models & Medical Data
*This cell pulls the Top 5 free models individually. If one fails, it tries a fallback.*

```python
# Ensure correct directory
%cd /content/LLM_KG

# Model list with fallbacks
models = [
    "llama3", 
    "mistral", 
    "gemma2", 
    "phi3", # Stable mini version (3.8B)
    "biomistral"
]

for model in models:
    print(f"📥 Pulling {model}...")
    result = os.system(f"ollama pull {model}")
    if result != 0:
        print(f"⚠️ Warning: Could not pull {model}. Trying fallback...")
        if model == "biomistral":
            !ollama pull medllama2
        elif model == "phi3":
            !ollama pull phi3:mini

# Download the ClinVec Knowledge Graph Data
!python3 src/ingestion/fetcher.py
print("\n✅ Setup Complete! Ready for Step 3.")
```

---

### 3️⃣ Step 3: Run Multi-Model Comparison
*This cell is now robust to model name changes.*

```python
import os
%cd /content/LLM_KG

from src.agents.graph import create_agentic_workflow
from src.agents.nodes import get_llm
from src.graph.builder import GraphBuilder
from src.ingestion.loader import load_clinical_notes
from tabulate import tabulate

# Load sample notes
notes_path = "data/raw/notes_sample.csv"
notes = load_clinical_notes(notes_path)[:3]
comparison_results = []

# List models that were successfully pulled
# (We check which ones are actually in the ollama library)
import subprocess
pulled_models_raw = subprocess.check_output(["ollama", "list"]).decode()
local_models = []
for m in ["llama3", "mistral", "gemma2", "phi3", "biomistral", "medllama2", "phi3:mini"]:
    if m in pulled_models_raw:
        local_models.append(m)

# Take only top 5 unique ones
local_models = list(dict.fromkeys(local_models))[:5]

print(f"🚀 Starting Benchmark on: {local_models}")
for model_name in local_models:
    print(f"  > Processing with {model_name}...")
    workflow = create_agentic_workflow()
    llm = get_llm("ollama", model_name)
    
    total_triples = 0
    for note in notes:
        state = {"clinical_note": note, "is_valid": False, "iterations": 0}
        try:
            final_state = workflow.invoke(state, config={"configurable": {"llm": llm}})
            total_triples += len(final_state["extracted_triples"])
        except Exception as e:
            print(f"    ❌ Error with {model_name}: {e}")
    
    comparison_results.append({
        "Model": model_name,
        "Total Triples": total_triples,
        "Avg Per Note": round(total_triples / len(notes), 2),
        "API Cost": "FREE"
    })

print("\n" + "="*50)
print("RESEARCH RESULTS: NO-API MODEL COMPARISON")
print("="*50)
print(tabulate(comparison_results, headers="keys", tablefmt="grid"))
```

---

### 4️⃣ Step 4: Visualise the Knowledge Graphs
```python
%cd /content/LLM_KG
from src.graph.visualizer import visualize_graph
from IPython.display import HTML, display

# Use the models from the previous step
for model_name in local_models:
    builder = GraphBuilder()
    llm = get_llm("ollama", model_name)
    workflow = create_agentic_workflow()
    
    # Generate KG for the first note
    print(f"Generating Visualization for {model_name}...")
    state = workflow.invoke({"clinical_note": notes[0]}, config={"configurable": {"llm": llm}})
    builder.add_triples(state["extracted_triples"])
    
    # Save visualization
    safe_name = model_name.replace(':', '_')
    output_path = f"data/processed/kg_{safe_name}.html"
    visualize_graph(builder.graph, output_path=output_path)
    
    print(f"\n🔍 Interactive KG: {model_name}")
    display(HTML(filename=output_path))
```
