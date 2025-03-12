"""
Graph-based Analysis for Shipping Fraud Detection

This module provides functions to build and analyze graphs from shipping data
to detect fraud rings and suspicious connections between entities.
"""

import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
import matplotlib.pyplot as plt
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ShippingFraudGraphAnalyzer:
    """
    Builds and analyzes graphs from shipping data to detect fraud rings.
    """
    
    def __init__(self):
        """
        Initialize the graph analyzer.
        """
        self.graph = None
        self.fraud_rings = None
        self.suspicious_nodes = None
        self.node_risk_scores = None
    
    def build_graph(self, 
                   df: pd.DataFrame, 
                   sender_id_col: str = 'SenderID',
                   recipient_address_col: str = 'RecipientAddressID',
                   device_col: str = 'DeviceInfo',
                   ip_col: str = 'ip_id',
                   email_col: str = 'SenderEmailDomain') -> nx.Graph:
        """
        Build a graph from shipping data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        sender_id_col : str
            Name of the sender ID column
        recipient_address_col : str
            Name of the recipient address ID column
        device_col : str
            Name of the device info column
        ip_col : str
            Name of the IP ID column
        email_col : str
            Name of the email domain column
            
        Returns:
        --------
        nx.Graph
            NetworkX graph representing connections between entities
        """
        logger.info("Building graph from shipping data...")
        
        # Create an empty undirected graph
        G = nx.Graph()
        
        # Add nodes and edges based on available columns
        available_cols = [col for col in [sender_id_col, recipient_address_col, device_col, ip_col, email_col] 
                         if col in df.columns]
        
        if not available_cols:
            logger.warning("No valid columns found for graph construction")
            return G
        
        # Add sender nodes
        if sender_id_col in available_cols:
            logger.info(f"Adding sender nodes from {sender_id_col}...")
            
            # Add sender nodes with attributes
            for _, row in df.iterrows():
                sender_id = row[sender_id_col]
                if pd.notna(sender_id):
                    # Add node if it doesn't exist
                    if not G.has_node(f"sender_{sender_id}"):
                        G.add_node(f"sender_{sender_id}", 
                                  type='sender', 
                                  id=sender_id,
                                  is_fraud=row.get('isFraud', 0) if 'isFraud' in df.columns else 0)
        
        # Add address nodes and connect to senders
        if recipient_address_col in available_cols and sender_id_col in available_cols:
            logger.info(f"Adding address nodes from {recipient_address_col} and connecting to senders...")
            
            for _, row in df.iterrows():
                sender_id = row[sender_id_col]
                address_id = row[recipient_address_col]
                
                if pd.notna(sender_id) and pd.notna(address_id):
                    # Add address node if it doesn't exist
                    if not G.has_node(f"address_{address_id}"):
                        G.add_node(f"address_{address_id}", 
                                  type='address', 
                                  id=address_id)
                    
                    # Connect sender to address
                    G.add_edge(f"sender_{sender_id}", 
                              f"address_{address_id}", 
                              weight=1)
        
        # Add device nodes and connect to senders
        if device_col in available_cols and sender_id_col in available_cols:
            logger.info(f"Adding device nodes from {device_col} and connecting to senders...")
            
            for _, row in df.iterrows():
                sender_id = row[sender_id_col]
                device_info = row[device_col]
                
                if pd.notna(sender_id) and pd.notna(device_info):
                    # Add device node if it doesn't exist
                    if not G.has_node(f"device_{device_info}"):
                        G.add_node(f"device_{device_info}", 
                                  type='device', 
                                  info=device_info)
                    
                    # Connect sender to device
                    G.add_edge(f"sender_{sender_id}", 
                              f"device_{device_info}", 
                              weight=1)
        
        # Add IP nodes and connect to senders
        if ip_col in available_cols and sender_id_col in available_cols:
            logger.info(f"Adding IP nodes from {ip_col} and connecting to senders...")
            
            for _, row in df.iterrows():
                sender_id = row[sender_id_col]
                ip_id = row[ip_col]
                
                if pd.notna(sender_id) and pd.notna(ip_id):
                    # Add IP node if it doesn't exist
                    if not G.has_node(f"ip_{ip_id}"):
                        G.add_node(f"ip_{ip_id}", 
                                  type='ip', 
                                  id=ip_id)
                    
                    # Connect sender to IP
                    G.add_edge(f"sender_{sender_id}", 
                              f"ip_{ip_id}", 
                              weight=1)
        
        # Add email nodes and connect to senders
        if email_col in available_cols and sender_id_col in available_cols:
            logger.info(f"Adding email nodes from {email_col} and connecting to senders...")
            
            for _, row in df.iterrows():
                sender_id = row[sender_id_col]
                email = row[email_col]
                
                if pd.notna(sender_id) and pd.notna(email):
                    # Add email node if it doesn't exist
                    if not G.has_node(f"email_{email}"):
                        G.add_node(f"email_{email}", 
                                  type='email', 
                                  domain=email)
                    
                    # Connect sender to email
                    G.add_edge(f"sender_{sender_id}", 
                              f"email_{email}", 
                              weight=1)
        
        logger.info(f"Graph construction complete. Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        self.graph = G
        return G
    
    def detect_fraud_rings(self, min_ring_size: int = 3, max_ring_size: int = 10) -> List[Set[str]]:
        """
        Detect potential fraud rings in the graph.
        
        Parameters:
        -----------
        min_ring_size : int
            Minimum size of a fraud ring
        max_ring_size : int
            Maximum size of a fraud ring
            
        Returns:
        --------
        List[Set[str]]
            List of detected fraud rings (sets of node IDs)
        """
        if self.graph is None:
            logger.warning("Graph not built. Call build_graph() first.")
            return []
        
        logger.info(f"Detecting fraud rings (size {min_ring_size}-{max_ring_size})...")
        
        # Get all sender nodes
        sender_nodes = [node for node, attr in self.graph.nodes(data=True) if attr.get('type') == 'sender']
        
        # Find connected components in the graph
        components = list(nx.connected_components(self.graph))
        logger.info(f"Found {len(components)} connected components")
        
        # Filter components by size
        potential_rings = []
        
        for component in components:
            # Count sender nodes in the component
            senders_in_component = [node for node in component if node in sender_nodes]
            
            # Check if the component has the right size
            if min_ring_size <= len(senders_in_component) <= max_ring_size:
                # Check if the component has shared resources (addresses, devices, IPs, emails)
                resource_nodes = component - set(senders_in_component)
                
                if resource_nodes:
                    # Check if there are shared resources among multiple senders
                    for resource in resource_nodes:
                        connected_senders = [node for node in self.graph.neighbors(resource) if node in sender_nodes]
                        
                        if len(connected_senders) >= 2:
                            # This resource is shared by multiple senders, consider it a potential ring
                            potential_rings.append(set(senders_in_component))
                            break
        
        logger.info(f"Detected {len(potential_rings)} potential fraud rings")
        
        self.fraud_rings = potential_rings
        return potential_rings
    
    def calculate_node_risk_scores(self) -> Dict[str, float]:
        """
        Calculate risk scores for nodes in the graph.
        
        Returns:
        --------
        Dict[str, float]
            Dictionary mapping node IDs to risk scores
        """
        if self.graph is None:
            logger.warning("Graph not built. Call build_graph() first.")
            return {}
        
        logger.info("Calculating node risk scores...")
        
        # Initialize risk scores
        risk_scores = {}
        
        # Calculate centrality measures
        degree_centrality = nx.degree_centrality(self.graph)
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        
        # Calculate risk scores based on centrality and node attributes
        for node, attr in self.graph.nodes(data=True):
            # Base risk score from centrality measures
            base_risk = 0.4 * degree_centrality.get(node, 0) + 0.6 * betweenness_centrality.get(node, 0)
            
            # Adjust risk based on node type
            node_type = attr.get('type', '')
            type_multiplier = 1.0
            
            if node_type == 'sender':
                # Higher risk for sender nodes with fraud flag
                is_fraud = attr.get('is_fraud', 0)
                type_multiplier = 2.0 if is_fraud else 1.0
            elif node_type == 'device':
                # Higher risk for certain device types
                device_info = attr.get('info', '')
                if isinstance(device_info, str) and ('android' in device_info.lower() or 'unknown' in device_info.lower()):
                    type_multiplier = 1.5
            elif node_type == 'email':
                # Higher risk for certain email domains
                email_domain = attr.get('domain', '')
                high_risk_domains = ['protonmail.com', 'tutanota.com', 'guerrillamail.com', 'temp-mail.org', 'mailinator.com']
                if isinstance(email_domain, str) and any(domain in email_domain.lower() for domain in high_risk_domains):
                    type_multiplier = 1.8
            
            # Calculate final risk score
            risk_scores[node] = base_risk * type_multiplier
        
        # Normalize risk scores to [0, 1]
        if risk_scores:
            max_risk = max(risk_scores.values())
            min_risk = min(risk_scores.values())
            
            if max_risk > min_risk:
                for node in risk_scores:
                    risk_scores[node] = (risk_scores[node] - min_risk) / (max_risk - min_risk)
        
        logger.info(f"Calculated risk scores for {len(risk_scores)} nodes")
        
        self.node_risk_scores = risk_scores
        return risk_scores
    
    def identify_suspicious_nodes(self, risk_threshold: float = 0.7) -> List[str]:
        """
        Identify suspicious nodes based on risk scores.
        
        Parameters:
        -----------
        risk_threshold : float
            Threshold for considering a node suspicious
            
        Returns:
        --------
        List[str]
            List of suspicious node IDs
        """
        if self.node_risk_scores is None:
            self.calculate_node_risk_scores()
        
        logger.info(f"Identifying suspicious nodes (risk threshold: {risk_threshold})...")
        
        # Filter nodes by risk score
        suspicious_nodes = [node for node, score in self.node_risk_scores.items() if score >= risk_threshold]
        
        logger.info(f"Identified {len(suspicious_nodes)} suspicious nodes")
        
        self.suspicious_nodes = suspicious_nodes
        return suspicious_nodes
    
    def visualize_graph(self, 
                       output_file: Optional[str] = None, 
                       highlight_rings: bool = True,
                       highlight_suspicious: bool = True) -> None:
        """
        Visualize the graph with highlighted fraud rings and suspicious nodes.
        
        Parameters:
        -----------
        output_file : Optional[str]
            Path to save the visualization. If None, display the plot.
        highlight_rings : bool
            Whether to highlight detected fraud rings
        highlight_suspicious : bool
            Whether to highlight suspicious nodes
        """
        if self.graph is None:
            logger.warning("Graph not built. Call build_graph() first.")
            return
        
        logger.info("Visualizing graph...")
        
        # Create a copy of the graph for visualization
        G = self.graph.copy()
        
        # Set up the plot
        plt.figure(figsize=(12, 10))
        
        # Define node colors based on type
        node_colors = []
        for node, attr in G.nodes(data=True):
            node_type = attr.get('type', '')
            
            if node_type == 'sender':
                color = 'blue'
            elif node_type == 'address':
                color = 'green'
            elif node_type == 'device':
                color = 'orange'
            elif node_type == 'ip':
                color = 'purple'
            elif node_type == 'email':
                color = 'red'
            else:
                color = 'gray'
            
            node_colors.append(color)
        
        # Define node sizes based on risk scores
        node_sizes = []
        if self.node_risk_scores:
            for node in G.nodes():
                risk_score = self.node_risk_scores.get(node, 0)
                # Scale size based on risk score (100-500)
                size = 100 + 400 * risk_score
                node_sizes.append(size)
        else:
            node_sizes = [300] * G.number_of_nodes()
        
        # Draw the graph
        pos = nx.spring_layout(G, seed=42)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
        nx.draw_networkx_edges(G, pos, alpha=0.3)
        
        # Highlight fraud rings if requested
        if highlight_rings and self.fraud_rings:
            for i, ring in enumerate(self.fraud_rings):
                ring_subgraph = G.subgraph(ring)
                nx.draw_networkx_edges(ring_subgraph, pos, width=2, edge_color=f'C{i % 10}', alpha=0.8)
        
        # Highlight suspicious nodes if requested
        if highlight_suspicious and self.suspicious_nodes:
            suspicious_nodes = [node for node in G.nodes() if node in self.suspicious_nodes]
            nx.draw_networkx_nodes(G, pos, nodelist=suspicious_nodes, node_color='red', node_size=[node_sizes[i] * 1.5 for i, node in enumerate(G.nodes()) if node in suspicious_nodes], alpha=0.8)
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Sender'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Address'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Device'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='IP'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Email')
        ]
        
        plt.legend(handles=legend_elements, loc='upper right')
        plt.title('Shipping Fraud Graph Analysis')
        plt.axis('off')
        
        # Save or display the plot
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            logger.info(f"Graph visualization saved to {output_file}")
        else:
            plt.show()
    
    def get_fraud_ring_report(self) -> pd.DataFrame:
        """
        Generate a report of detected fraud rings.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing information about detected fraud rings
        """
        if self.fraud_rings is None:
            logger.warning("Fraud rings not detected. Call detect_fraud_rings() first.")
            return pd.DataFrame()
        
        logger.info("Generating fraud ring report...")
        
        # Prepare report data
        report_data = []
        
        for i, ring in enumerate(self.fraud_rings):
            # Get sender nodes in the ring
            sender_nodes = [node for node in ring if self.graph.nodes[node].get('type') == 'sender']
            
            # Get shared resources
            shared_resources = defaultdict(list)
            
            for sender in sender_nodes:
                for neighbor in self.graph.neighbors(sender):
                    neighbor_type = self.graph.nodes[neighbor].get('type', '')
                    
                    if neighbor_type != 'sender':
                        shared_resources[neighbor_type].append(neighbor)
            
            # Count shared resources by type
            shared_counts = {resource_type: len(set(resources)) for resource_type, resources in shared_resources.items()}
            
            # Calculate average risk score
            avg_risk = 0
            if self.node_risk_scores:
                sender_risks = [self.node_risk_scores.get(sender, 0) for sender in sender_nodes]
                avg_risk = sum(sender_risks) / len(sender_risks) if sender_risks else 0
            
            # Add to report
            report_data.append({
                'RingID': i + 1,
                'Size': len(sender_nodes),
                'AvgRiskScore': avg_risk,
                'SharedAddresses': shared_counts.get('address', 0),
                'SharedDevices': shared_counts.get('device', 0),
                'SharedIPs': shared_counts.get('ip', 0),
                'SharedEmails': shared_counts.get('email', 0),
                'SenderIDs': ', '.join([node.split('_')[1] for node in sender_nodes])
            })
        
        # Create DataFrame
        report_df = pd.DataFrame(report_data)
        
        # Sort by risk score
        if 'AvgRiskScore' in report_df.columns:
            report_df = report_df.sort_values('AvgRiskScore', ascending=False)
        
        logger.info(f"Generated report for {len(report_df)} fraud rings")
        
        return report_df


if __name__ == "__main__":
    # Example usage
    from src.data.data_loader import ShippingFraudDataLoader
    
    # Load and preprocess data
    data_loader = ShippingFraudDataLoader()
    train_data, _ = data_loader.preprocess_data()
    
    # Build and analyze graph
    graph_analyzer = ShippingFraudGraphAnalyzer()
    graph_analyzer.build_graph(train_data)
    
    # Detect fraud rings
    fraud_rings = graph_analyzer.detect_fraud_rings()
    
    # Calculate risk scores
    risk_scores = graph_analyzer.calculate_node_risk_scores()
    
    # Identify suspicious nodes
    suspicious_nodes = graph_analyzer.identify_suspicious_nodes()
    
    # Generate fraud ring report
    report = graph_analyzer.get_fraud_ring_report()
    
    print("\nFraud Ring Report:")
    print(report.head())
    
    # Visualize graph
    graph_analyzer.visualize_graph(output_file='fraud_graph.png') 