from pyvis.network import Network
import networkx as nx
import os

def visualize_graph(nx_graph, output_path="data/processed/kg_visualization.html"):
    """
    Visualize a NetworkX graph using Pyvis and save as HTML.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a Pyvis network
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=False, directed=True)
    
    # Load the NetworkX graph
    for node, attrs in nx_graph.nodes(data=True):
        net.add_node(node, label=node, title=node)
        
    for source, target, data in nx_graph.edges(data=True):
        net.add_edge(source, target, label=data.get('label', ''), title=f"Confidence: {data.get('confidence', 'N/A')}")
        
    # Set physics options for better layout
    net.toggle_physics(True)
    
    # Save the visualization
    net.save_graph(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    # Test visualization
    test_graph = nx.DiGraph()
    test_graph.add_edge("Aspirin", "Inflammation", label="TREATS", confidence=0.9)
    visualize_graph(test_graph)
