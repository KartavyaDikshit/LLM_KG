import os
from neo4j import GraphDatabase
from src.agents.state import Triple

class Neo4jManager:
    """Manages Neo4j operations for the Knowledge Graph."""

    def __init__(self, uri: str = None, user: str = None, password: str = None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        self.driver.close()

    def clear_database(self):
        """Wipes the entire database for a fresh start."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("Neo4j database cleared.")

    def upload_triples(self, triples: list, domain: str = "medical"):
        """
        Uploads a list of triples to Neo4j.
        Deduplicates nodes and relationships using MERGE.
        """
        with self.driver.session() as session:
            for triple in triples:
                # Sanitize predicate to be valid Cypher relationship type (no spaces, uppercase)
                clean_pred = str(triple.predicate).replace(" ", "_").upper()
                # Ensure it's not empty or just numbers
                if not clean_pred or clean_pred[0].isdigit():
                    clean_pred = "REL_" + clean_pred if clean_pred else "RELATED_TO"

                query = (
                    "MERGE (s:Entity {name: $sub}) "
                    "ON CREATE SET s.domain = $domain "
                    "MERGE (o:Entity {name: $obj}) "
                    "ON CREATE SET o.domain = $domain "
                    "WITH s, o "
                    f"MERGE (s)-[r:{clean_pred}]->(o) "
                    "SET r.confidence = $conf"
                )
                session.run(query, sub=triple.subject, obj=triple.obj, domain=domain, conf=triple.confidence)
        print(f"Uploaded {len(triples)} triples to Neo4j.")

    def get_summary(self):
        """Returns basic stats about the graph."""
        with self.driver.session() as session:
            nodes = session.run("MATCH (n) RETURN count(n) as count").single()["count"]
            rels = session.run("MATCH ()-[r]->() RETURN count(r) as count").single()["count"]
            return {"nodes": nodes, "relationships": rels}
