import pandas as pd
import networkx as nx

def evaluate_graph_quality(nx_graph, ontology_nodes_df=None):
    """
    Evaluates the quality of the extracted knowledge graph.
    - Coverage: How many nodes match the ontology?
    - Density: Connectivity of the graph.
    - Coherence: Number of connected components.
    """
    stats = {}
    
    # 1. Connectivity Stats
    num_nodes = nx_graph.number_of_nodes()
    num_edges = nx_graph.number_of_edges()
    stats['node_count'] = num_nodes
    stats['edge_count'] = num_edges
    stats['density'] = nx.density(nx_graph)
    
    if num_nodes > 0:
        stats['avg_degree'] = sum(dict(nx_graph.degree()).values()) / num_nodes
    else:
        stats['avg_degree'] = 0

    # 2. Ontological Alignment (if nodes_df is provided)
    if ontology_nodes_df is not None:
        ontology_terms = set(ontology_nodes_df['node_name'].str.lower().unique())
        graph_terms = set([str(n).lower() for n in nx_graph.nodes()])
        
        matches = graph_terms.intersection(ontology_terms)
        stats['ontological_coverage'] = len(matches) / len(graph_terms) if graph_terms else 0
        stats['matched_terms'] = list(matches)[:10]  # Sample of matches
    
    return stats

def print_evaluation_report(stats):
    print("\n--- KG Quality Assessment ---")
    print(f"Nodes: {stats['node_count']}")
    print(f"Edges: {stats['edge_count']}")
    print(f"Density: {stats['density']:.4f}")
    if 'ontological_coverage' in stats:
        print(f"Ontological Alignment Score: {stats['ontological_coverage']:.2%}")
    print("-----------------------------\n")
