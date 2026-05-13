import networkx as nx
import json
import os

class GraphBuilder:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_triples(self, triples):
        """Add a list of Triple objects to the graph."""
        for triple in triples:
            self.graph.add_edge(
                triple.subject,
                triple.obj,
                label=triple.predicate,
                confidence=triple.confidence
            )

    def save_graph(self, file_path="data/processed/knowledge_graph.json"):
        """Save the graph structure to a JSON file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        data = nx.node_link_data(self.graph)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Graph saved to {file_path}")

    def load_graph(self, file_path="data/processed/knowledge_graph.json"):
        """Load the graph structure from a JSON file."""
        if not os.path.exists(file_path):
            return
        with open(file_path, 'r') as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data)
