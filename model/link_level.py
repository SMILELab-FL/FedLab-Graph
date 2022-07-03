# Codes below are copied from
# https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/model/link_level.py

import torch
from torch_geometric.data import Data

from model.mlp import MLP
from model.gcn import GCN_Net
from model.sage import SAGE_Net
from model.gat import GAT_Net
from model.gin import GIN_Net
from model.gpr import GPR_Net


class GNN_Net_Link(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 hidden=64,
                 max_depth=2,
                 dropout=.0,
                 gnn='gcn',
                 layers=2):
        r"""GNN model with LinkPredictor for link prediction tasks.

        Arguments:
            in_channels (int): input channels.
            out_channels (int): output channels.
            hidden (int): hidden dim for all modules.
            max_depth (int): number of layers for gnn.
            dropout (float): dropout probability.
            gnn (str): name of gnn type, use ("gcn" or "gin").
            layers (int): number of layers for LinkPredictor.

        """
        super(GNN_Net_Link, self).__init__()
        self.dropout = dropout

        # GNN layer
        if gnn == 'gcn':
            self.gnn = GCN_Net(in_channels=in_channels,
                               out_channels=hidden,
                               hidden=hidden,
                               max_depth=max_depth,
                               dropout=dropout)
        elif gnn == 'sage':
            self.gnn = SAGE_Net(in_channels=in_channels,
                                out_channels=hidden,
                                hidden=hidden,
                                max_depth=max_depth,
                                dropout=dropout)
        elif gnn == 'gat':
            self.gnn = GAT_Net(in_channels=in_channels,
                               out_channels=hidden,
                               hidden=hidden,
                               max_depth=max_depth,
                               dropout=dropout)
        elif gnn == 'gin':
            self.gnn = GIN_Net(in_channels=in_channels,
                               out_channels=hidden,
                               hidden=hidden,
                               max_depth=max_depth,
                               dropout=dropout)
        elif gnn == 'gpr':
            self.gnn = GPR_Net(in_channels=in_channels,
                               out_channels=hidden,
                               hidden=hidden,
                               K=max_depth,
                               dropout=dropout)
        else:
            raise ValueError(f'Unsupported gnn type: {gnn}.')

        dim_list = [hidden for _ in range(layers)]
        self.output = MLP([hidden] + dim_list + [out_channels],
                          batch_norm=True)

    def forward(self, data):
        if isinstance(data, Data):
            x, edge_index = data.x, data.edge_index
        elif isinstance(data, tuple):
            x, edge_index = data
        else:
            raise TypeError('Unsupported data type!')

        x = self.gnn((x, edge_index))
        return x

    def link_predictor(self, x, edge_index):
        x = x[edge_index[0]] * x[edge_index[1]]
        x = self.output(x)
        return x
