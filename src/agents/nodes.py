import os
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from src.agents.state import AgentState, Triple, BaseModel, Field
from pydantic import ValidationError
from langchain_core.runnables import Runnable

# Mock LLM for local testing without API key
class MockLLM(Runnable):
    def invoke(self, input, config=None):
        content = "Mock Strategy: Identify medical entities and their relationships."
        return type('obj', (object,), {'content': content})

    def with_structured_output(self, schema):
        return MockStructuredLLM(schema)

class MockStructuredLLM(Runnable):
    def __init__(self, schema):
        self.schema = schema
    def invoke(self, inputs, config=None):
        note = inputs.get('note', "").lower()
        triples = []
        if "aspirin" in note:
            triples.append(Triple(subject="Aspirin", predicate="TREATS", obj="Pain/Inflammation", confidence=0.99))
        if "lisinopril" in note:
            triples.append(Triple(subject="Lisinopril", predicate="TREATS", obj="Hypertension", confidence=0.99))
        if "infarction" in note:
            triples.append(Triple(subject="Myocardial Infarction", predicate="DIAGNOSIS", obj="Acute", confidence=0.95))
        if "sepsis" in note:
            triples.append(Triple(subject="Sepsis", predicate="CONDITION", obj="Severe", confidence=0.95))
        if "cancer" in note:
            triples.append(Triple(subject="Breast Cancer", predicate="DIAGNOSIS", obj="Stage II", confidence=0.98))
        if "paclitaxel" in note:
            triples.append(Triple(subject="Paclitaxel", predicate="TREATS", obj="Breast Cancer", confidence=0.99))
        
        # Fallback if no keywords matched
        if not triples:
            triples.append(Triple(subject="Patient", predicate="EXHIBITS", obj="Clinical Symptoms", confidence=0.5))
            
        return type('obj', (object,), {'triples': triples})

def get_llm(model_type="gemini", model_name=None):
    """Factory to get the requested LLM. Supports Gemini, Groq, and Ollama."""
    if os.getenv("MOCK_MODE", "false").lower() == "true":
        return MockLLM()
    
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
        return MockLLM()

def planner_node(state: AgentState, config=None):
    """Analyze the clinical note and determine the extraction focus."""
    # Retrieve LLM from config if provided, else default
    llm_instance = config.get("configurable", {}).get("llm", get_llm()) if config else get_llm()
    
    prompt = ChatPromptTemplate.from_template(
        "You are an expert medical knowledge engineer. Analyze the following clinical note.\n"
        "1. Identify ALL medications, dosages, and routes.\n"
        "2. Identify ALL diagnoses, symptoms, and chronic conditions.\n"
        "3. Identify ALL procedures and lab tests mentioned.\n"
        "4. Map the relationships between them (e.g., Drug TREATS Disease, Test DIAGNOSES Condition).\n\n"
        "Note: {note}\n\n"
        "Provide a detailed, structured extraction strategy to ensure maximum density of the resulting graph."
    )
    chain = prompt | llm_instance
    strategy = chain.invoke({"note": state["clinical_note"]})
    
    return {
        "planner_strategy": strategy.content,
        "iterations": state.get("iterations", 0) + 1
    }

def extractor_node(state: AgentState, config=None):
    """Extract triples from clinical text using the planner's strategy."""
    llm_instance = config.get("configurable", {}).get("llm", get_llm()) if config else get_llm()
    structured_llm = llm_instance.with_structured_output(ExtractionOutput)
    
    prompt = ChatPromptTemplate.from_template(
        "You are a clinical NLP extractor. Your goal is to extract a HIGH DENSITY medical knowledge graph.\n"
        "Use the strategy provided to pull EVERY possible Subject-Predicate-Object triple.\n\n"
        "Strategy: {strategy}\n"
        "Note: {note}\n\n"
        "Guidelines:\n"
        "- Be exhaustive. Do not miss any relationships mentioned.\n"
        "- Use standard medical terminology for nodes.\n"
        "- Predicates should be clear and consistent (e.g., HAS_SYMPTOM, PRESCRIBED_FOR, CONTRAINDICATED_WITH).\n\n"
        "If this is a re-extraction, address this feedback: {feedback}\n"
    )
    
    feedback = state.get("validation_feedback", "None")
    result = structured_llm.invoke({
        "strategy": state["planner_strategy"],
        "note": state["clinical_note"],
        "feedback": feedback
    })
    
    return {"extracted_triples": result.triples}

def validator_node(state: AgentState, config=None):
    """Validate the extracted triples against logical consistency and medical truth."""
    if os.getenv("MOCK_MODE", "false").lower() == "true":
        return {"is_valid": True, "validation_feedback": None}
        
    llm_instance = config.get("configurable", {}).get("llm", get_llm()) if config else get_llm()
    
    triples_text = "\n".join([f"{t.subject} - {t.predicate} -> {t.obj}" for t in state["extracted_triples"]])
    
    prompt = ChatPromptTemplate.from_template(
        "You are a medical ontology validator. Evaluate these extracted triples for accuracy.\n"
        "Triples:\n{triples}\n\n"
        "Original Note:\n{note}\n\n"
        "Check for:\n"
        "1. Accuracy: Does the note actually support this relationship?\n"
        "2. Direction: Is the relationship direction correct?\n"
        "3. Specificity: Are the terms medically precise?\n\n"
        "If they are excellent, reply 'PASSED'. Otherwise, provide corrective instructions for the extractor."
    )
    
    chain = prompt | llm_instance
    evaluation = chain.invoke({
        "triples": triples_text,
        "note": state["clinical_note"]
    })
    
    is_valid = "PASSED" in evaluation.content.upper()
    return {
        "is_valid": is_valid,
        "validation_feedback": None if is_valid else evaluation.content
    }
