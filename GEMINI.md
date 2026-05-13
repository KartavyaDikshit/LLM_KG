# Project Instructions: Personalized Medicine KG

## Overview
This project builds a Knowledge Graph (KG) for personalized medicine using Agentic GraphRAG with LangChain/LangGraph.

## Standards & Conventions
- **Language**: Python 3.10+
- **Style**: PEP 8
- **Agents**: Use LangGraph for stateful multi-agent workflows.
- **Data**: 
  - Clinical notes follow MIMIC-IV schema (primarily `NOTEEVENTS` equivalent).
  - Ontology is based on ClinVec (Harvard Zitnik Lab).
- **Validation**: All extractions MUST be validated against the ontology before insertion.

## Workflow
1. Ingest clinical notes and ontology.
2. Run Agentic Pipeline (Planner -> Extractor -> Validator).
3. Construct NetworkX graph from validated triples.
4. Visualize using Pyvis.
