import json
import os
from typing import List, Dict, Any
from src.agents.state import Triple

class MilestoneManager:
    def __init__(self, file_path="data/processed/milestone.json"):
        self.file_path = file_path
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w') as f:
                json.dump([], f)

    def save(self, index: int, triples: List[Triple], metadata: Dict[str, Any] = None):
        """Appends to a local JSON list."""
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            data = []
        
        entry = {
            "index": index,
            "triples": [t.model_dump() if hasattr(t, "model_dump") else t.dict() for t in triples],
            "metadata": metadata or {}
        }
        data.append(entry)
        
        with open(self.file_path, 'w') as f:
            json.dump(data, f, indent=4)

    def get_processed_indices(self) -> List[int]:
        """Returns a list of indices already processed."""
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
            return [item["index"] for item in data]
        except (json.JSONDecodeError, FileNotFoundError):
            return []

    def get_all_triples(self) -> List[Triple]:
        """Aggregates everything for visualization."""
        try:
            with open(self.file_path, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
        
        all_triples = []
        for item in data:
            for t in item["triples"]:
                all_triples.append(Triple(**t))
        return all_triples
