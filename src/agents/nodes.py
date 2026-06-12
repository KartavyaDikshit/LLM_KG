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
    """Factory to get the local Ollama LLM. External APIs removed."""
    try:
        name = model_name or "llama3"
        return ChatOllama(model=name, temperature=0)
    except Exception as e:
        print(f"LLM Initialization Error: {e}. Falling back to default Ollama (llama3).")
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

def robust_json_extract(text: str) -> Any:
    """The most robust parser for handling conversational LLM outputs."""
    # 1. Clean common conversational hallucinations inside numeric fields (e.g. 0.8 (Inferred...))
    text = re.sub(r'(\d+\.?\d*)\s*\([^)]*\)', r'\1', text)
    
    # 2. Extract potential JSON candidates
    # Find all blocks in markdown
    blocks = re.findall(r'```(?:json)?\s*(.*?)\s*```', text, re.DOTALL)
    # Also consider the whole text as a candidate
    candidates = blocks + [text]
    
    for cand in candidates:
        # Find outermost braces/brackets
        s_brace, e_brace = cand.find('{'), cand.rfind('}')
        s_bracket, e_bracket = cand.find('['), cand.rfind(']')
        
        if s_brace == -1 and s_bracket == -1: continue
        
        # Decide which one is the actual JSON start
        start = s_brace if (s_bracket == -1 or (s_brace != -1 and s_brace < s_bracket)) else s_bracket
        end = e_brace if (e_bracket == -1 or (e_brace != -1 and e_brace > e_bracket)) else e_bracket
        
        clean = cand[start:end+1]
        
        # Strategy A: Standard JSON
        try:
            return json.loads(clean)
        except:
            pass
            
        # Strategy B: Clean common LLM formatting errors (single quotes, trailing commas)
        try:
            # Replace single quotes with double quotes
            # Note: This is naive but works for simple cases. 
            # Better to use ast.literal_eval for single-quoted structures.
            import ast
            return ast.literal_eval(clean)
        except:
            pass
            
        # Strategy C: Final attempt - Replace newline/tabs in strings that break JSON
        try:
            fixed = clean.replace('\n', ' ').replace('\r', ' ')
            return json.loads(fixed)
        except:
            pass
            
    return None

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
        "- Allowed Relations: {allowed_predicates}\n\n"
        "CRITICAL: Output MUST be a SINGLE JSON object only. NO PREAMBLE. NO APOLOGIES. NO EXPLANATIONS.\n"
        "Format: {{'triples': [{{'subject': '...', 'predicate': '...', 'obj': '...', 'confidence': 1.0}}]}}\n"
        "Feedback to incorporate: {feedback}\n"
    )
    
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
            "feedback": feedback
        })
        
        content = response.content
        triples = []
        data = robust_json_extract(content)
        
        if data:
            result_list = data.get('triples', []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            for t in result_list:
                if isinstance(t, dict):
                    # Robust field mapping
                    s = str(t.get('subject') or 'Unknown')
                    p = str(t.get('predicate') or 'RELATED_TO')
                    o = str(t.get('obj') or 'Unknown')
                    c = t.get('confidence')
                    try:
                        triples.append(Triple(
                            subject=s,
                            predicate=p,
                            obj=o,
                            confidence=float(c if c is not None else 1.0)
                        ))
                    except: pass
                    
        if not triples:
            print(f"  [Debug] Parser failed to find triples in LLM output. Raw snippet: {content[:200]}...")
            
        return {"extracted_triples": triples}
    except Exception as e:
        print(f"Extractor Node Error: {e}")
        return {"extracted_triples": []}

def validator_node(state: AgentState, config=None):
    """Validate extracted triples against domain constraints."""
    domain_cfg = load_domain_config(state.get("domain", "medical"))
    llm = config.get("configurable", {}).get("llm", get_llm()) if config else get_llm()
    
    extracted = state.get("extracted_triples", [])
    if not extracted:
        return {"is_valid": True, "validation_feedback": "Valid (empty)."}

    triples_text = "\n".join([f"{t.subject} - {t.predicate} -> {t.obj}" for t in extracted])
    
    prompt = ChatPromptTemplate.from_template(
        "You are a {domain_name} ontology validator.\n"
        "Review these extracted triples: {triples}\n"
        "Input Text: {note}\n\n"
        "If all are grounded in text and valid, reply 'PASSED'. Otherwise provide feedback."
    )
    
    try:
        chain = prompt | llm
        evaluation = chain.invoke({
            "domain_name": domain_cfg["domain_name"],
            "triples": triples_text,
            "note": state.get("input_text", "")
        })
        
        is_valid = "PASSED" in evaluation.content.upper()
        return {
            "is_valid": is_valid,
            "validation_feedback": None if is_valid else evaluation.content
        }
    except Exception as e:
        return {"is_valid": True, "validation_feedback": None}

def deduplicator_node(state: AgentState, config=None):
    """Normalize and deduplicate entities."""
    llm = config.get("configurable", {}).get("llm", get_llm()) if config else get_llm()
    extracted = state.get("extracted_triples", [])
    if not extracted:
        return {"extracted_triples": []}

    triples_text = "\n".join([f"{t.subject} - {t.predicate} -> {t.obj}" for t in extracted])
    
    prompt = ChatPromptTemplate.from_template(
        "Normalize entity names in these triples for consistency: {triples}\n"
        "Output the cleaned triples as JSON with key 'triples'."
    )
    
    try:
        chain = prompt | llm
        response = chain.invoke({"triples": triples_text})
        data = robust_json_extract(response.content)
        
        cleaned_triples = []
        if data:
            result_list = data.get('triples', []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            for t in result_list:
                if isinstance(t, dict):
                    s = str(t.get('subject') or 'Unknown')
                    p = str(t.get('predicate') or 'RELATED_TO')
                    o = str(t.get('obj') or 'Unknown')
                    c = t.get('confidence')
                    try:
                        cleaned_triples.append(Triple(
                            subject=s,
                            predicate=p,
                            obj=o,
                            confidence=float(c if c is not None else 1.0)
                        ))
                    except: pass
        
        return {"extracted_triples": cleaned_triples if cleaned_triples else extracted}
    except Exception as e:
        return {"extracted_triples": extracted}

from langchain_community.graphs import Neo4jGraph
from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

def query_node(state: AgentState, config=None):
    """Query the Neo4j graph."""
    llm = config.get("configurable", {}).get("llm", get_llm()) if config else get_llm()
    query = state.get("query")
    if not query: return {"answer": "No query provided."}
    
    try:
        graph = Neo4jGraph(
            url=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", "password")
        )
        chain = GraphCypherQAChain.from_llm(llm=llm, graph=graph, verbose=True, allow_dangerous_requests=True)
        response = chain.invoke({"query": query})
        return {"answer": response.get("result", "I couldn't find an answer.")}
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}
