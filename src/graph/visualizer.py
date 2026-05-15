from pyvis.network import Network
import networkx as nx
import os

def visualize_graph(nx_graph, output_path="data/processed/kg_visualization.html"):
    """
    Visualize a NetworkX graph with categorical coloring and physics.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Improved visualization settings
    net = Network(height="600px", width="100%", bgcolor="#ffffff", font_color="#333333", notebook=False, directed=True)
    
    # Define color scheme for medical categories
    color_map = {
        "drug": "#3b82f6",      # Blue
        "medication": "#3b82f6",
        "disease": "#ef4444",   # Red
        "condition": "#ef4444",
        "diagnosis": "#ef4444",
        "symptom": "#f59e0b",   # Orange
        "test": "#10b981",      # Green
        "procedure": "#8b5cf6", # Purple
        "patient": "#6b7280"    # Gray
    }
    
    # Add nodes with smart coloring
    for node in nx_graph.nodes():
        node_lower = str(node).lower()
        color = "#94a3b8" # Default Gray-blue
        
        for key, val in color_map.items():
            if key in node_lower:
                color = val
                break
        
        net.add_node(node, label=str(node), title=str(node), color=color, size=25)
        
    # Add edges with labels
    for source, target, data in nx_graph.edges(data=True):
        net.add_edge(source, target, label=data.get('label', ''), 
                     title=f"Relation: {data.get('label', '')}",
                     width=2, color="#cbd5e1", arrows="to")
        
    # Set sophisticated physics for better spread
    net.set_options("""
    var options = {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 100,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": { "iterations": 150 }
      }
    }
    """)
    
    net.save_graph(output_path)

if __name__ == "__main__":
    # Test visualization
    test_graph = nx.DiGraph()
    test_graph.add_edge("Aspirin", "Inflammation", label="TREATS", confidence=0.9)
    visualize_graph(test_graph)
