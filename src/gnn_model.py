import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import networkx as nx
from typing import Dict, Any, Tuple
import random

class GraphSAGEAnomalyDetector(torch.nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.classifier = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # 2-layer GraphSAGE
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)
        
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        
        # Output probability of being an anomaly
        out = self.classifier(x)
        return torch.sigmoid(out)

def create_synthetic_training_data() -> Data:
    """
    Creates a mock transaction graph containing a mix of normal topologies 
    and a few hidden circular trading rings (anomalies) for training.
    """
    # 50 nodes
    num_nodes = 50
    # Basic features: in-degree, out-degree, total_amount_in, total_amount_out
    x = torch.zeros((num_nodes, 4), dtype=torch.float)
    
    edge_list = []
    y = torch.zeros(num_nodes, dtype=torch.long)
    
    # Normal random connections
    for _ in range(80):
        src = random.randint(0, num_nodes - 10)
        dst = random.randint(0, num_nodes - 10)
        if src != dst:
            edge_list.append([src, dst])
            
    # Inject circular trading rings (Nodes 40-49)
    # Ring 1
    edge_list.extend([[40, 41], [41, 42], [42, 43], [43, 40]])
    y[40:44] = 1 # Mark as anomalous
    
    # Ring 2
    edge_list.extend([[45, 46], [46, 47], [47, 48], [48, 49], [49, 45]])
    y[45:50] = 1 # Mark as anomalous
    
    # Feature imputation (mocking feature engineering)
    for src, dst in edge_list:
        x[src, 1] += 1.0 # out-degree
        x[src, 3] += random.uniform(10.0, 100.0) # total_amount_out
        x[dst, 0] += 1.0 # in-degree
        x[dst, 2] += random.uniform(10.0, 100.0) # total_amount_in
        
    # row normalization
    x = F.normalize(x, p=2, dim=1)
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    data = Data(x=x, edge_index=edge_index, y=y)
    return data

def train_gnn_model() -> GraphSAGEAnomalyDetector:
    """
    Trains the GraphSAGE model on synthetic anomalous data and returns the trained model.
    """
    data = create_synthetic_training_data()
    model = GraphSAGEAnomalyDetector(in_channels=4, hidden_channels=16, out_channels=1)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.BCELoss()
    
    model.train()
    for _ in range(100):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out.squeeze(), data.y.float())
        loss.backward()
        optimizer.step()
        
    return model

# Global singleton model for prototype
_MODEL_INSTANCE = None

def get_gnn_model() -> GraphSAGEAnomalyDetector:
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        _MODEL_INSTANCE = train_gnn_model()
    return _MODEL_INSTANCE

def score_graph_with_gnn(G: nx.DiGraph) -> Tuple[float, list]:
    """
    Extracts features from the NetworkX DiGraph, passes them through the trained GraphSAGE model,
    and returns a maximum graph risk score and a list of high-risk entity nodes.
    """
    if G.number_of_nodes() == 0:
        return 0.0, []
        
    # Mapping node names to integer IDs
    node_to_id = {node: i for i, node in enumerate(G.nodes())}
    id_to_node = {i: node for node, i in node_to_id.items()}
    num_nodes = len(node_to_id)
    
    # Build edge index
    edge_list = []
    for u, v in G.edges():
        edge_list.append([node_to_id[u], node_to_id[v]])
        
    if not edge_list:
        return 0.0, []
        
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    
    # Extract features: in-degree, out-degree, total_amount_in, total_amount_out
    x = torch.zeros((num_nodes, 4), dtype=torch.float)
    in_deg = G.in_degree()
    out_deg = G.out_degree()
    
    for u, v, data in G.edges(data=True):
        amt = data.get("amount", 0.0)
        u_id, v_id = node_to_id[u], node_to_id[v]
        x[u_id, 3] += amt
        x[v_id, 2] += amt
        
    for node, i in node_to_id.items():
        x[i, 0] = in_deg[node]
        x[i, 1] = out_deg[node]
        
    x = F.normalize(x, p=2, dim=1)
    
    model = get_gnn_model()
    model.eval()
    
    with torch.no_grad():
        out = model(x, edge_index).squeeze().tolist()
        
    if num_nodes == 1:
        out = [out]
        
    # Aggregate scores (e.g. max risk in graph)
    max_score = float(max(out)) if out else 0.0
    
    # Find high-risk nodes (score > 0.5)
    risky_nodes = []
    for i, score in enumerate(out):
        if score > 0.5:
            risky_nodes.append(id_to_node[i])
            
    return max_score, risky_nodes
