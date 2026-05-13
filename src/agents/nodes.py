import os
from typing import List
from langchain_google_genai import ChatGoogleGenerativeAI
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

# Determine if we should use Mock Mode
USE_MOCK = os.getenv("MOCK_MODE", "false").lower() == "true" or not os.getenv("GEMINI_API_KEY")

if USE_MOCK:
    print("!!! RUNNING IN MOCK MODE (No API Key required) !!!")
    llm = MockLLM()
else:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)

class ExtractionOutput(BaseModel):
    triples: List[Triple] = Field(description="List of medical triples extracted from text")

def planner_node(state: AgentState):
    """Analyze the clinical note and determine the extraction focus."""
    prompt = ChatPromptTemplate.from_template(
        "You are a medical knowledge graph planner. Analyze the following clinical note and "
        "determine the key entities (diseases, drugs, procedures) and the relationships "
        "that should be extracted to align with a medical ontology. \n\n"
        "Note: {note}\n\n"
        "Provide a concise extraction strategy."
    )
    chain = prompt | llm
    strategy = chain.invoke({"note": state["clinical_note"]})
    
    return {
        "planner_strategy": strategy.content,
        "iterations": state.get("iterations", 0) + 1
    }

def extractor_node(state: AgentState):
    """Extract triples from clinical text using the planner's strategy."""
    structured_llm = llm.with_structured_output(ExtractionOutput)
    
    prompt = ChatPromptTemplate.from_template(
        "You are a clinical information extractor. Use the provided strategy to pull "
        "Subject-Predicate-Object triples from the clinical note. \n\n"
        "Strategy: {strategy}\n"
        "Note: {note}\n\n"
        "If this is a re-extraction, address the feedback: {feedback}\n\n"
        "Format the output as a list of triples."
    )
    
    feedback = state.get("validation_feedback", "None")
    result = structured_llm.invoke({
        "strategy": state["planner_strategy"],
        "note": state["clinical_note"],
        "feedback": feedback
    })
    
    return {"extracted_triples": result.triples}

def validator_node(state: AgentState):
    """Validate the extracted triples against logical consistency and medical truth."""
    if USE_MOCK:
        return {
            "is_valid": True,
            "validation_feedback": None
        }
        
    # In a full implementation, this would query the ClinVec CSVs/DB.
    # For the prototype, we use the LLM as a medical critic primed with ontology concepts.
    
    triples_text = "\n".join([f"{t.subject} - {t.predicate} -> {t.obj}" for t in state["extracted_triples"]])
    
    prompt = ChatPromptTemplate.from_template(
        "You are a medical ontology validator. Evaluate the following extracted triples "
        "for clinical accuracy and logical consistency. \n\n"
        "Triples:\n{triples}\n\n"
        "Original Note:\n{note}\n\n"
        "Check for:\n"
        "1. Logical direction (e.g., Treatment -> Disease, not Disease -> Treatment)\n"
        "2. Entity grounding (Are the entities real medical terms?)\n"
        "3. Hallucinations (Is the relationship supported by the note?)\n\n"
        "Provide 'PASSED' if all are valid, otherwise provide detailed feedback for correction."
    )
    
    chain = prompt | llm
    evaluation = chain.invoke({
        "triples": triples_text,
        "note": state["clinical_note"]
    })
    
    is_valid = "PASSED" in evaluation.content.upper()
    return {
        "is_valid": is_valid,
        "validation_feedback": None if is_valid else evaluation.content
    }
