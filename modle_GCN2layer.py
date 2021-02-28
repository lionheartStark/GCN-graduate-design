import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding
from torch.nn import Parameter
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric_temporal.nn.recurrent import EvolveGCNO
from torch_geometric.utils.convert import to_networkx
from torch_geometric.utils import to_undirected


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels, conv1=GCNConv, conv2=GCNConv, use_skip=False):
        super(GCN, self).__init__()
        self.conv1 = conv1(num_node_features, hidden_channels[0])
        self.conv2 = conv2(hidden_channels[0], 2)
        self.use_skip = use_skip
        if self.use_skip:
            self.weight = nn.init.xavier_normal_(Parameter(torch.Tensor(num_node_features, 2)))

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, data.edge_index)
        if self.use_skip:
            x = F.softmax(x + torch.matmul(data.x, self.weight), dim=-1)
        else:
            x = F.softmax(x, dim=-1)
        return x

    def embed(self, data):
        x = self.conv1(data.x, data.edge_index)
        return x
