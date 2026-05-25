import os
import json
import re
import yaml
from typing import List, Dict, Any
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.agents.state import AgentState, Triple, BaseModel, Field
from pydantic import ValidationError
from langchain_core.runnables import Runnable

class ExtractionOutput(BaseModel):
    triples: List[Triple] = Field(description="List of triples extracted from text")

def load_domain_config(domain: str) -> Dict[str, Any]:
    """Load domain configuration from YAML."""
    config_path = f"src/config/domains/{domain}.yaml"
    if not os.path.exists(config_path):
        # Fallback to medical if domain not found
        config_path = "src/config/domains/medical.yaml"
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_llm(model_type="gemini", model_name=None):
    """Factory to get the requested LLM."""
    if model_type == "gemini":
        name = model_name or "gemini-1.5-flash"
        return ChatGoogleGenerativeAI(model=name, temperature=0)
    elif model_type == "groq":
        name = model_name or "llama-3.1-70b-versatile"
        return ChatGroq(model=name, temperature=0)
    elif model_type == "ollama":
        name = model_name or "llama3"
        return ChatOllama(model=name, temperature=0)
    else:
        return ChatOllama(model="llama3", temperature=0)

def planner_node(state: AgentState, config=None):
    """Analyze the input text and determine the extraction focus based on domain."""
    domain_cfg = load_domain_config(state.get("domain", "medical"))
    llm = config.get("configurable", {}).get("llm", get_llm()) if config else get_llm()
    
    prompt = ChatPromptTemplate.from_template(
        "You are an expert knowledge graph engineer specializing in the {domain_name} domain.\n"
        "Domain Description: {description}\n"
        "Instruction: {planner_instruction}\n"
        "Entity Types: {entity_types}\n"
        "Allowed Relations: {allowed_predicates}\n\n"
        "Input Text: {note}\n\n"
        "Provide a detailed, structured extraction strategy to ensure maximum density of the resulting graph."
    )
    
    chain = prompt | llm
    strategy = chain.invoke({
        "domain_name": domain_cfg["domain_name"],
        "description": domain_cfg["description"],
        "planner_instruction": domain_cfg["planner_instruction"],
        "entity_types": domain_cfg["entity_types"],
        "allowed_predicates": domain_cfg["allowed_predicates"],
        "note": state["input_text"]
    })
    
    return {
        "planner_strategy": strategy.content,
        "iterations": state.get("iterations", 0) + 1
    }

def extractor_node(state: AgentState, config=None):
    """Extract triples from text using the planner's strategy and domain constraints."""
    domain_cfg = load_domain_config(state.get("domain", "medical"))
    llm = config.get("configurable", {}).get("llm", get_llm()) if config else get_llm()
        
    prompt = ChatPromptTemplate.from_template(
        "You are a high-precision knowledge extractor for the {domain_name} domain.\n"
        "Goal: {extractor_instruction}\n"
        "Strategy: {strategy}\n"
        "Input Text: {note}\n\n"
        "Constraints:\n"
        "- Entity Types: {entity_types}\n"
        "- Allowed Relations: {allowed_predicates}\n"
        "- Be exhaustive. Do not miss any relationships mentioned.\n\n"
        "Output MUST be a valid JSON object with a key 'triples' containing a list of objects with 'subject', 'predicate', 'obj', and 'confidence' keys.\n"
        "Example: {{\"triples\": [{{\"subject\": \"Entity A\", \"predicate\": \"RELATION\", \"obj\": \"Entity B\", \"confidence\": 1.0}}]}}\n"
        "If this is a re-extraction, address this feedback: {feedback}\n"
    )
    
    chain = prompt | llm
    
    feedback = state.get("validation_feedback", "None")
    response = chain.invoke({
        "domain_name": domain_cfg["domain_name"],
        "extractor_instruction": domain_cfg["extractor_instruction"],
        "strategy": state["planner_strategy"],
        "note": state["input_text"],
        "entity_types": domain_cfg["entity_types"],
        "allowed_predicates": domain_cfg["allowed_predicates"],
        "feedback": feedback
    })
    
    content = response.content
    triples = []
    try:
        match = re.search(r'({.*}|\[.*\])', content, re.DOTALL)
        if match:
            raw_json = match.group(0)
            data = json.loads(raw_json)
            result_list = data.get('triples', []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
                
            for t in result_list:
                try:
                    triples.append(Triple(**t))
                except:
                    continue
    except Exception as e:
        print(f"Warning: Failed to parse JSON: {e}")
    
    return {"extracted_triples": triples}

def validator_node(state: AgentState, config=None):
    """Validate extracted triples against domain constraints."""
    domain_cfg = load_domain_config(state.get("domain", "medical"))
    llm = config.get("configurable", {}).get("llm", get_llm()) if config else get_llm()
    
    triples_text = "\n".join([f"{t.subject} - {t.predicate} -> {t.obj}" for t in state["extracted_triples"]])
    
    prompt = ChatPromptTemplate.from_template(
        "You are a {domain_name} ontology validator.\n"
        "Evaluate these extracted triples against the domain rules and input text.\n\n"
        "Domain Constraints:\n"
        "- Entity Types: {entity_types}\n"
        "- Allowed Relations: {allowed_predicates}\n\n"
        "Triples:\n{triples}\n\n"
        "Original Text:\n{note}\n\n"
        "Check for:\n"
        "1. Accuracy: Does the text support this relationship?\n"
        "2. Compliance: Do entities and predicates match the allowed types/relations?\n"
        "3. Specificity: Are the terms precise?\n\n"
        "If they are excellent, reply 'PASSED'. Otherwise, provide corrective instructions for the extractor."
    )
    
    chain = prompt | llm
    evaluation = chain.invoke({
        "domain_name": domain_cfg["domain_name"],
        "entity_types": domain_cfg["entity_types"],
        "allowed_predicates": domain_cfg["allowed_predicates"],
        "triples": triples_text,
        "note": state["input_text"]
    })
    
    is_valid = "PASSED" in evaluation.content.upper()
    return {
        "is_valid": is_valid,
        "validation_feedback": None if is_valid else evaluation.content
    }
