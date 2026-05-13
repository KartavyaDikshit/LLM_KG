# User Guide: Building Your Medical Knowledge Graph

## 1. Setup
Install the required dependencies:
```bash
pip install -r requirements.txt
```

Set your Gemini API key in a `.env` file:
```text
GEMINI_API_KEY=your_actual_api_key_here
```

## 2. Prepare Data
- **Ontology**: Run the fetcher to download ClinVec data from Harvard Dataverse:
  ```bash
  python -m src.ingestion.fetcher
  ```
- **Clinical Notes**: Place your MIMIC-IV `NOTEEVENTS.csv` (or a similar CSV with a `text` column) in `data/raw/`.

## 3. Run the Pipeline
Run the main script to process clinical notes and build the KG:
```bash
python src/main.py --notes data/raw/your_notes.csv --limit 10
```

### Options:
- `--fetch-ontology`: Download ClinVec data before starting.
- `--limit`: Number of notes to process (default: 5).

## 4. Explore Results
- **JSON Graph**: `data/processed/knowledge_graph.json` contains the raw graph data.
- **Interactive Visualization**: Open `data/processed/kg_visualization.html` in any web browser to explore the interactive network map.

## 5. Directory Structure
- `data/raw/`: Input clinical notes.
- `data/ontology/`: ClinVec CSV files.
- `data/processed/`: Generated KG and visualizations.
- `src/agents/`: LangGraph agent definitions.
- `src/graph/`: KG construction and visualization logic.
