# Architectural Blueprint: Agentic GraphRAG for Personalized Medicine

This document outlines the architectural decisions and data flow for the Personalized Medicine Knowledge Graph (LLM-KG) project.

## 1. High-Level Architecture
The system follows an **Agentic GraphRAG** (Retrieval-Augmented Generation) pattern. Unlike traditional RAG, which relies on vector similarity, GraphRAG leverages the structured relationships of a Knowledge Graph to provide multi-hop reasoning and deterministic clinical grounding.

### Layers:
- **Ingestion Layer**: Handles the raw inputs. It fetches established medical truth (ClinVec) and ingests unstructured patient data (MIMIC-IV).
- **Agentic Pipeline (LangGraph)**: A stateful multi-agent system that iteratively refines medical extractions.
- **Graph Layer**: Maintains the persistent Knowledge Graph using NetworkX.
- **Visualization Layer**: Renders the graph into interactive network maps.

## 2. Agentic Workflow
We use **LangGraph** to manage the interaction between specialized AI agents.

### Agents:
1.  **Planner Agent**:
    - *Role*: Strategist.
    - *Task*: Analyzes the complexity of a clinical note and determines which medical domains (e.g., oncology, cardiology) and relationship types (e.g., `TREATS`, `CAUSES`, `CONTRAINDICATED`) are most relevant.
2.  **Extractor Agent (Gemini 1.5 Pro)**:
    - *Role*: Parser.
    - *Task*: Performs Named Entity Recognition (NER) and Relation Extraction (RE). It outputs structured triples `(Subject, Predicate, Object)` with confidence scores.
3.  **Validator Agent (Critic)**:
    - *Role*: Quality Assurance.
    - *Task*: Cross-references the Extractor's output against the clinical note and the ClinVec ontology. If inaccuracies are found, it provides specific feedback for re-extraction.

## 3. Data Integration
- **MIMIC-IV**: The primary source of clinical text. The system extracts patient-specific facts.
- **ClinVec**: The "Ground Truth" schema. It contains over 5 million relationships across biological scales, ensuring that the LLM aligns its extractions with standardized medical terminology.

## 4. Graph Structure
The resulting graph is a **Directed Multigraph**.
- **Nodes**: Medical entities (Drugs, Diseases, Genes, Symptoms).
- **Edges**: Verified relationships with associated confidence scores.

## 5. Performance Advantages
- **Scalability**: Automated extraction replaces manual expert coding.
- **Accuracy**: The Validator loop significantly reduces hallucinations.
- **Explainability**: Every edge in the graph can be traced back to the original clinical note and the supporting ontology.
