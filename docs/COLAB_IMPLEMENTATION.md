# Section 2: Implementation Patch
Run this cell to apply the Multi-Domain architecture. This writes the refactored nodes and domain configurations to the file system.

```python
import os

# Create Domain Config Directory
os.makedirs('src/config/domains', exist_ok=True)

# 1. Write Medical Config
with open('src/config/domains/medical.yaml', 'w') as f:
    f.write("""
domain_name: "medical"
description: "Clinical notes and medical relationships"
entity_types: ["Drug", "Disease", "Symptom", "Procedure"]
allowed_predicates: ["TREATS", "CAUSES", "DIAGNOSES", "HAS_SYMPTOM"]
planner_instruction: "Identify medications, dosages, conditions, and symptoms."
extractor_instruction: "Extract a high-density medical knowledge graph."
color_map:
  drug: "#3b82f6"
  disease: "#ef4444"
  symptom: "#f59e0b"
  patient: "#6b7280"
""")

# 2. Write Legal Config
with open('src/config/domains/legal.yaml', 'w') as f:
    f.write("""
domain_name: "legal"
description: "Commercial contracts and legal obligations"
entity_types: ["Party", "Agreement", "Obligation", "Jurisdiction"]
allowed_predicates: ["BOUND_BY", "GOVERNED_BY", "MANDATES"]
planner_instruction: "Identify parties, their obligations, and governing laws."
extractor_instruction: "Extract a legal knowledge graph focusing on who owes what."
color_map:
  party: "#1e293b"
  agreement: "#0f172a"
  obligation: "#dc2626"
  jurisdiction: "#059669"
""")

# 3. Apply Multi-Domain Node Logic
patch_code = r'''
import os, json, re, yaml
try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.agents.state import AgentState, Triple

def load_domain_config(domain):
    path = f"src/config/domains/{domain}.yaml"
    with open(path, "r") as f: return yaml.safe_load(f)

def get_llm(model_name="llama3"):
    return ChatOllama(model=model_name, temperature=0)

def planner_node(state, config=None):
    cfg = load_domain_config(state.get("domain", "medical"))
    llm = config.get("configurable", {}).get("llm", get_llm()) if config else get_llm()
    prompt = ChatPromptTemplate.from_template("Domain: {name}\nGoal: {instr}\nText: {text}")
    res = (prompt | llm).invoke({"name": cfg["domain_name"], "instr": cfg["planner_instruction"], "text": state["input_text"]})
    return {"planner_strategy": res.content, "iterations": state.get("iterations", 0) + 1}

def extractor_node(state, config=None):
    cfg = load_domain_config(state.get("domain", "medical"))
    llm = config.get("configurable", {}).get("llm", get_llm()) if config else get_llm()
    prompt = ChatPromptTemplate.from_template("Extract JSON triples for {name}.\nStrategy: {strat}\nText: {text}\nFormat: {'triples': [{'subject': '', 'predicate': '', 'obj': '', 'confidence': 1.0}]}")
    res = (prompt | llm).invoke({"name": cfg["domain_name"], "strat": state["planner_strategy"], "text": state["input_text"]})
    triples = []
    match = re.search(r"(\{.*\}|\[.*\])", res.content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            items = data.get("triples", []) if isinstance(data, dict) else data
            for i in items: triples.append(Triple(subject=str(i.get("subject", "Unknown")), predicate=str(i.get("predicate", "RELATED_TO")), obj=str(i.get("obj", "Unknown")), confidence=1.0))
        except: pass
    return {"extracted_triples": triples}

def validator_node(state, config=None):
    return {"is_valid": True}
'''

with open('src/agents/nodes.py', 'w') as f: f.write(patch_code)

# 4. Update Visualizer for Multi-Domain
vis_patch = r'''
from pyvis.network import Network
import os, yaml

def visualize_graph(nx_graph, domain="medical", output_path="data/processed/kg.html"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    net = Network(height="550px", width="100%", bgcolor="#ffffff", directed=True)
    cfg_path = f"src/config/domains/{domain}.yaml"
    color_map = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f: color_map = yaml.safe_load(f).get("color_map", {})

    for node in nx_graph.nodes():
        color = "#94a3b8"
        for k, v in color_map.items():
            if k in str(node).lower(): color = v; break
        net.add_node(node, label=str(node), color=color)
    for s, t, d in nx_graph.edges(data=True):
        net.add_edge(s, t, label=d.get("label", ""))
    net.save_graph(output_path)
'''
with open('src/graph/visualizer.py', 'w') as f: f.write(vis_patch)

print("✅ Multi-Domain Implementation Patched!")
```
