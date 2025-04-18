import torch
from torch import nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

num_atom_type = 119
num_chirality_tag = 3

num_bond_type = 5
num_bond_direction = 3
num_lep_feature = 1

class GINEConv(MessagePassing):
    def __init__(self, emb_dim, edge_emb_dim=None):
        super(GINEConv, self).__init__()

        if edge_emb_dim is None:
            edge_emb_dim = emb_dim

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2*emb_dim),
            nn.ReLU(),
            nn.Linear(2*emb_dim, emb_dim)
        )

        self.edge_embedding1 = nn.Embedding(num_bond_type, edge_emb_dim)
        self.edge_embedding2 = nn.Embedding(num_bond_direction, edge_emb_dim)
        nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.lep_linear = nn.Linear(num_lep_feature, edge_emb_dim)

        self.edge_encoder_mlp = nn.Sequential(
            nn.Linear(edge_emb_dim * 3, edge_emb_dim),
            nn.ReLU(),
            nn.Linear(edge_emb_dim, emb_dim)
        )

    def forward(self, x, edge_index, edge_attr):
        if edge_attr.size(1) == 2:
            zero_lep = torch.zeros(edge_attr.size(0), 1, dtype=torch.float, device=edge_attr.device)
            edge_attr = torch.cat([edge_attr.float(), zero_lep], dim=1)
        elif edge_attr.size(1) != 3:
             raise ValueError(f"Edge attributes have unexpected dimension: {edge_attr.size(1)}. Expected 2 or 3.")

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        self_loop_attr = torch.zeros(x.size(0), 3, device=edge_attr.device)
        self_loop_attr[:, 0] = 4
        self_loop_attr[:, 1] = 0
        self_loop_attr[:, 2] = 0.0
        self_loop_attr = self_loop_attr.to(edge_attr.dtype)

        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_type = edge_attr[:, 0].long()
        edge_dir = edge_attr[:, 1].long()
        edge_lep = edge_attr[:, 2].unsqueeze(-1)

        type_emb = self.edge_embedding1(edge_type)
        dir_emb = self.edge_embedding2(edge_dir)
        lep_feat = self.lep_linear(edge_lep)

        combined_edge_features = torch.cat([type_emb, dir_emb, lep_feat], dim=1)
        edge_embeddings = self.edge_encoder_mlp(combined_edge_features)

        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


class GINet(nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat
    Output:
        node representations
    """
    def __init__(self, num_layer=5, emb_dim=300, feat_dim=256, drop_ratio=0, pool='mean'):
        super(GINet, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.feat_dim = feat_dim
        self.drop_ratio = drop_ratio

        self.x_embedding1 = nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = nn.Embedding(num_chirality_tag, emb_dim)
        nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        # List of MLPs
        self.gnns = nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINEConv(emb_dim=emb_dim))

        # List of batchnorms
        self.batch_norms = nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))
        
        if pool == 'mean':
            self.pool = global_mean_pool
        elif pool == 'max':
            self.pool = global_max_pool
        elif pool == 'add':
            self.pool = global_add_pool
        
        self.feat_lin = nn.Linear(self.emb_dim, self.feat_dim)

        self.out_lin = nn.Sequential(
            nn.Linear(self.feat_dim, self.feat_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(self.feat_dim, self.feat_dim//2)
        )

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        h = self.x_embedding1(x[:,0]) + self.x_embedding2(x[:,1])

        for layer in range(self.num_layer):
            h = self.gnns[layer](h, edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

        h = self.pool(h, data.batch)
        h = self.feat_lin(h)
        out = self.out_lin(h)
        
        return h, out

    def load_my_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                print(f"Warning: Ignoring parameter '{name}' from checkpoint not found in current model.")
                continue
            if isinstance(param, nn.parameter.Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except Exception as e:
                print(f"Warning: Could not load parameter '{name}'. Error: {e}")