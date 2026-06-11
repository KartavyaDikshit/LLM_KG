import os
import json
import re
import yaml
from typing import List, Dict, Any
try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.agents.state import AgentState, Triple, BaseModel, Field
from pydantic import ValidationError

def load_domain_config(domain: str) -> Dict[str, Any]:
    """Load domain configuration from YAML."""
    config_path = f"src/config/domains/{domain}.yaml"
    if not os.path.exists(config_path):
        config_path = "src/config/domains/medical.yaml"
    
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {
            "domain_name": "general",
            "description": "General knowledge extraction",
            "entity_types": [],
            "allowed_predicates": [],
            "planner_instruction": "Extract all relationships.",
            "extractor_instruction": "Extract triples.",
            "validator_instruction": "Validate triples."
        }

def get_llm(model_type="ollama", model_name=None):
    """Factory to get the requested LLM with lazy imports."""
    try:
        if model_type == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            name = model_name or "gemini-1.5-flash"
            return ChatGoogleGenerativeAI(model=name, temperature=0)
        elif model_type == "groq":
            from langchain_groq import ChatGroq
            name = model_name or "llama-3.1-70b-versatile"
            return ChatGroq(model=name, temperature=0)
        elif model_type == "ollama":
            name = model_name or "llama3"
            return ChatOllama(model=name, temperature=0)
        else:
            return ChatOllama(model="llama3", temperature=0)
    except Exception as e:
        print(f"LLM Initialization Error: {e}. Falling back to default Ollama.")
        return ChatOllama(model="llama3", temperature=0)

def planner_node(state: AgentState, config=None):
    """Analyze the input text and determine the extraction focus based on domain."""
    domain_cfg = load_domain_config(state.get("domain", "medical"))
    llm = config.get("configurable", {}).get("llm", get_llm()) if config else get_llm()
    
    # Robust: No literal braces in template
    prompt = ChatPromptTemplate.from_template(
        "You are an expert knowledge graph engineer specializing in the {domain_name} domain.\n"
        "Domain Description: {description}\n"
        "Instruction: {planner_instruction}\n"
        "Entity Types: {entity_types}\n"
        "Allowed Relations: {allowed_predicates}\n\n"
        "Input Text: {note}\n\n"
        "Provide a detailed, structured extraction strategy to ensure maximum density of the resulting graph."
    )
    
    try:
        chain = prompt | llm
        strategy = chain.invoke({
            "domain_name": domain_cfg["domain_name"],
            "description": domain_cfg["description"],
            "planner_instruction": domain_cfg["planner_instruction"],
            "entity_types": str(domain_cfg["entity_types"]),
            "allowed_predicates": str(domain_cfg["allowed_predicates"]),
            "note": state.get("input_text", "")
        })
        return {
            "planner_strategy": strategy.content,
            "iterations": state.get("iterations", 0) + 1
        }
    except Exception as e:
        print(f"Planner Node Error: {e}")
        return {"planner_strategy": "Exhaustive extraction.", "iterations": state.get("iterations", 0) + 1}

def extractor_node(state: AgentState, config=None):
    """Extract triples from text using the planner's strategy and domain constraints."""
    domain_cfg = load_domain_config(state.get("domain", "medical"))
    llm = config.get("configurable", {}).get("llm", get_llm()) if config else get_llm()
        
    # Robust: JSON example passed as a variable to avoid brace collision
    prompt = ChatPromptTemplate.from_template(
        "You are a high-precision knowledge extractor for the {domain_name} domain.\n"
        "Goal: {extractor_instruction}\n"
        "Strategy: {strategy}\n"
        "Input Text: {note}\n\n"
        "Constraints:\n"
        "- Entity Types: {entity_types}\n"
        "- Allowed Relations: {allowed_predicates}\n"
        "- Be exhaustive. Do not miss any relationships mentioned.\n\n"
        "Format Instructions: {format_instr}\n"
        "If this is a re-extraction, address this feedback: {feedback}\n"
    )
    
    json_example = 'Output MUST be valid JSON with key "triples" containing a list of objects with "subject", "predicate", "obj", and "confidence". Example: {"triples": [{"subject": "A", "predicate": "B", "obj": "C", "confidence": 1.0}]}'
    
    try:
        chain = prompt | llm
        feedback = state.get("validation_feedback") or "None"
        response = chain.invoke({
            "domain_name": domain_cfg["domain_name"],
            "extractor_instruction": domain_cfg["extractor_instruction"],
            "strategy": state.get("planner_strategy", ""),
            "note": state.get("input_text", ""),
            "entity_types": str(domain_cfg["entity_types"]),
            "allowed_predicates": str(domain_cfg["allowed_predicates"]),
            "format_instr": json_example,
            "feedback": feedback
        })
        
        content = response.content
        triples = []
        
        # Robust parsing
        def extract_json_data(text):
            m = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
            if m: text = m.group(1)
            start, end = text.find('{'), text.rfind('}')
            if start != -1 and end != -1:
                try: return json.loads(text[start:end+1].replace("'", '"'))
                except: pass
                import ast
                try: return ast.literal_eval(text[start:end+1])
                except: pass
            start, end = text.find('['), text.rfind(']')
            if start != -1 and end != -1:
                try: return json.loads(text[start:end+1].replace("'", '"'))
                except: pass
                import ast
                try: return ast.literal_eval(text[start:end+1])
                except: pass
            return None
            
        data = extract_json_data(content)
        if data:
            result_list = data.get('triples', []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            for t in result_list:
                if isinstance(t, dict):
                    triples.append(Triple(
                        subject=str(t.get('subject', 'Unknown')),
                        predicate=str(t.get('predicate', 'RELATED_TO')),
                        obj=str(t.get('obj', 'Unknown')),
                        confidence=float(t.get('confidence', 0.8))
                    ))
        if not triples:
            print(f"  [Debug] LLM output parsing failed or 0 triples found. Raw output: {content.strip()}")
        return {"extracted_triples": triples}
    except Exception as e:
        print(f"Extractor Node Error: {e}")
        return {"extracted_triples": []}

def validator_node(state: AgentState, config=None):
    """Validate extracted triples against domain constraints and check for hallucinations."""
    domain_cfg = load_domain_config(state.get("domain", "medical"))
    llm = config.get("configurable", {}).get("llm", get_llm()) if config else get_llm()
    
    extracted = state.get("extracted_triples", [])
    if not extracted:
        # If the LLM legitimately found no triples, accept it and move on instead of looping 3 times
        return {"is_valid": True, "validation_feedback": "Valid (empty)."}

    triples_text = "\n".join([f"{t.subject} - {t.predicate} -> {t.obj}" for t in extracted])
    
    prompt = ChatPromptTemplate.from_template(
        "You are a {domain_name} ontology validator.\n"
        "Review the following extracted triples based on the input text and domain ontology.\n\n"
        "Input Text: {note}\n"
        "Ontology Entity Types: {entity_types}\n"
        "Ontology Allowed Relations: {allowed_predicates}\n\n"
        "Extracted Triples:\n{triples}\n\n"
        "Task:\n"
        "1. Identify any triples that are NOT supported by the text (hallucinations).\n"
        "2. Identify any triples that violate the ontology constraints.\n"
        "3. If all triples are valid and grounded in text, reply exactly with 'PASSED'.\n"
        "4. Otherwise, provide specific feedback for the extractor to fix the errors."
    )
    
    try:
        chain = prompt | llm
        evaluation = chain.invoke({
            "domain_name": domain_cfg["domain_name"],
            "entity_types": str(domain_cfg["entity_types"]),
            "allowed_predicates": str(domain_cfg["allowed_predicates"]),
            "triples": triples_text,
            "note": state.get("input_text", "")
        })
        
        is_valid = "PASSED" in evaluation.content.upper()
        return {
            "is_valid": is_valid,
            "validation_feedback": None if is_valid else evaluation.content
        }
    except Exception as e:
        print(f"Validator Node Error: {e}")
        return {"is_valid": True, "validation_feedback": None}

def deduplicator_node(state: AgentState, config=None):
    """Normalize and deduplicate entities in the extracted triples."""
    llm = config.get("configurable", {}).get("llm", get_llm()) if config else get_llm()
    extracted = state.get("extracted_triples", [])
    if not extracted:
        return {"extracted_triples": []}

    triples_text = "\n".join([f"{t.subject} - {t.predicate} -> {t.obj}" for t in extracted])
    
    prompt = ChatPromptTemplate.from_template(
        "You are a knowledge graph entity normalizer.\n"
        "Input Triples:\n{triples}\n\n"
        "Task:\n"
        "1. Normalize entity names (e.g., 'T2D' -> 'Type 2 Diabetes', 'MI' -> 'Myocardial Infarction').\n"
        "2. Ensure consistent casing and naming conventions.\n"
        "3. Output the cleaned triples in the same JSON format.\n\n"
        "Format Instructions: {format_instr}"
    )
    
    json_example = 'Output MUST be valid JSON with key "triples" containing a list of objects with "subject", "predicate", "obj", and "confidence".'
    
    try:
        chain = prompt | llm
        response = chain.invoke({
            "triples": triples_text,
            "format_instr": json_example
        })
        
        content = response.content
        cleaned_triples = []
        
        # Robust parsing
        def extract_json_data(text):
            m = re.search(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
            if m: text = m.group(1)
            start, end = text.find('{'), text.rfind('}')
            if start != -1 and end != -1:
                try: return json.loads(text[start:end+1].replace("'", '"'))
                except: pass
            start, end = text.find('['), text.rfind(']')
            if start != -1 and end != -1:
                try: return json.loads(text[start:end+1].replace("'", '"'))
                except: pass
            return None
            
        data = extract_json_data(content)
        if data:
            result_list = data.get('triples', []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            for t in result_list:
                if isinstance(t, dict):
                    cleaned_triples.append(Triple(
                        subject=str(t.get('subject', 'Unknown')),
                        predicate=str(t.get('predicate', 'RELATED_TO')),
                        obj=str(t.get('obj', 'Unknown')),
                        confidence=float(t.get('confidence', 0.8))
                    ))
        
        return {"extracted_triples": cleaned_triples}
    except Exception as e:
        print(f"Deduplicator Node Error: {e}")
        return {"extracted_triples": extracted}

from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

def query_node(state: AgentState, config=None):
    """Query the Neo4j graph using natural language and return an answer."""
    llm = config.get("configurable", {}).get("llm", get_llm()) if config else get_llm()
    query = state.get("query")
    
    if not query:
        return {"answer": "No query provided."}
    
    try:
        # Initialize graph connection
        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password")
        )
        
        # Build QA Chain
        chain = GraphCypherQAChain.from_llm(
            llm=llm,
            graph=graph,
            verbose=True,
            allow_dangerous_requests=True # Required for Cypher execution in newer LC versions
        )
        
        response = chain.invoke({"query": query})
        return {"answer": response.get("result", "I couldn't find an answer in the graph.")}
    except Exception as e:
        print(f"Query Node Error: {e}")
        return {"answer": f"Error querying graph: {str(e)}"}
