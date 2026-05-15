from pyvis.network import Network
import networkx as nx
import os

def visualize_graph(nx_graph, output_path="data/processed/kg_visualization.html"):
    """
    Visualize a NetworkX graph with categorical coloring and CDN-only assets for Colab.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Use a white background for better visibility in Colab
    net = Network(height="550px", width="100%", bgcolor="#ffffff", font_color="#333333", notebook=False, directed=True)
    
    # Force CDN to avoid local file path issues in Colab
    net.set_edge_smooth('dynamic')
    
    color_map = {
        "drug": "#3b82f6", "medication": "#3b82f6",
        "disease": "#ef4444", "condition": "#ef4444",
        "symptom": "#f59e0b", "test": "#10b981",
        "procedure": "#8b5cf6", "patient": "#6b7280"
    }
    
    for node in nx_graph.nodes():
        node_label = str(node)
        color = "#94a3b8"
        for key, val in color_map.items():
            if key in node_label.lower():
                color = val
                break
        net.add_node(node, label=node_label, title=node_label, color=color, size=30)
        
    for source, target, data in nx_graph.edges(data=True):
        label = data.get('label', '')
        net.add_edge(source, target, label=label, title=label, width=2, color="#cbd5e1", arrows="to")
        
    # High-performance physics
    net.set_options("""
    var options = {
      "physics": {
        "barnesHut": { "gravitationalConstant": -2000, "centralGravity": 0.3, "springLength": 95, "springConstant": 0.04 },
        "minVelocity": 0.75
      }
    }
    """)
    
    # Save with no local assets
    net.save_graph(output_path)

if __name__ == "__main__":
    # Test visualization
    test_graph = nx.DiGraph()
    test_graph.add_edge("Aspirin", "Inflammation", label="TREATS", confidence=0.9)
    visualize_graph(test_graph)
