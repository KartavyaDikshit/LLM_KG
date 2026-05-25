# Section 3: Execution & Benchmarking
Run this cell last to process medical and legal inputs and generate interactive Knowledge Graphs.

```python
import os, base64
from src.agents.graph import create_agentic_workflow
from src.agents.nodes import get_llm
from src.graph.builder import GraphBuilder
from src.graph.visualizer import visualize_graph
from IPython.display import HTML, display

def render_colab_kg(path, title):
    if not os.path.exists(path): return
    with open(path, 'r', encoding='utf-8') as f: html = f.read()
    b64 = base64.b64encode(html.encode('utf-8')).decode('utf-8')
    iframe = f'<div style="border:1px solid #ddd; border-radius:8px; padding:10px; margin-bottom:20px;"><h3>🔍 {title}</h3><iframe src="data:text/html;base64,{b64}" width="100%" height="600px" style="border:none;"></iframe></div>'
    display(HTML(iframe))

workflow = create_agentic_workflow()
llm = get_llm("llama3")

# --- TEST 1: MEDICAL ---
print("🔬 Running Medical Extraction...")
med_text = "Patient with acute myocardial infarction prescribed Aspirin and Lisinopril."
res_med = workflow.invoke({"input_text": med_text, "domain": "medical", "is_valid": False, "iterations": 0}, config={"configurable": {"llm": llm}})
builder_med = GraphBuilder()
builder_med.add_triples(res_med["extracted_triples"])
visualize_graph(builder_med.graph, domain="medical", output_path="data/processed/med_kg.html")
render_colab_kg("data/processed/med_kg.html", "Medical KG (Llama 3)")

# --- TEST 2: LEGAL ---
print("⚖️ Running Legal Extraction...")
legal_text = "ACME Corp is bound by the laws of Delaware. This Agreement mandates payment to Beta Inc by June 1st."
res_leg = workflow.invoke({"input_text": legal_text, "domain": "legal", "is_valid": False, "iterations": 0}, config={"configurable": {"llm": llm}})
builder_leg = GraphBuilder()
builder_leg.add_triples(res_leg["extracted_triples"])
visualize_graph(builder_leg.graph, domain="legal", output_path="data/processed/legal_kg.html")
render_colab_kg("data/processed/legal_kg.html", "Legal KG (Llama 3)")
```
