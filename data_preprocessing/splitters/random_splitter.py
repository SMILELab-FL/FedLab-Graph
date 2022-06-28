import torch

from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx, from_networkx
from fedlab.utils.dataset.functional import lognormal_unbalance_split, dirichlet_unbalance_split

import numpy as np
import networkx as nx

EPSILON = 1e-5


class RandomSplitter(BaseTransform):
    r"""
    Split Data into small data via random sampling.

    Args:
        client_num (int): Split data into client_num of pieces.
        random_type (int): Random sampling method. Accepted: 0 for ``"lognormal"`` and others for `"dirichlet"`
        split_param (float): Param for the corresponding ``random_type`` sampling method.
        sum_rate (float): Sum of samples of the unique nodes for each client, default to 1
        overlapping_rate(float): Additional samples of overlapping data, eg. '0.4'
        drop_edge(float): Drop edges (drop_edge / client_num) for each client whthin overlapping part.
        
    """
    def __init__(self,
                 client_num,
                 random_type=0,
                 split_param=0,
                 sum_rate=1,
                 overlapping_rate=0,
                 drop_edge=0):

        self.random_type = random_type
        self.split_param = split_param
        self.sum_rate = sum_rate
        self.ovlap = overlapping_rate
        if abs((sum_rate + self.ovlap) - 1) > EPSILON:
            raise ValueError(
                f'The sum of sampling_rate:{self.sum_rate} and overlapping_rate({self.ovlap}) should be 1.'
            )

        self.client_num = client_num
        self.drop_edge = drop_edge

    def __call__(self, data):
        data.index_orig = torch.arange(data.num_nodes)
        G = to_networkx(
            data,
            node_attrs=['x', 'y', 'train_mask', 'val_mask', 'test_mask'],
            to_undirected=True)
        nx.set_node_attributes(G,
                               dict([(nid, nid)
                                     for nid in range(nx.number_of_nodes(G))]),
                               name="index_orig")
        client_node_idx = {idx: [] for idx in range(self.client_num)}

        indices = np.random.permutation(round(self.sum_rate * data.num_nodes))

        if self.random_type == 0:
            self.client_sample_nums = lognormal_unbalance_split(self.client_num, data.num_nodes, unbalance_sgm=self.split_param)
        else:
            self.client_sample_nums = dirichlet_unbalance_split(self.client_num, data.num_nodes, alpha=self.split_param)

        node_count = 0
        for idx, sample_num in enumerate(self.client_sample_nums):
            client_node_idx[idx] = indices[node_count: node_count+sample_num]
            node_count += sample_num

        if self.ovlap:
            ovlap_nodes = indices[round(self.sum_rate * data.num_nodes):]
            for idx in client_node_idx:
                client_node_idx[idx] = np.concatenate(
                    (client_node_idx[idx], ovlap_nodes))

        # Drop_edge index for each client
        if self.drop_edge:
            ovlap_graph = nx.Graph(nx.subgraph(G, ovlap_nodes))
            ovlap_edge_ind = np.random.permutation(
                ovlap_graph.number_of_edges())
            drop_all = ovlap_edge_ind[:round(ovlap_graph.number_of_edges() *
                                             self.drop_edge)]
            drop_client = [
                drop_all[s:s + round(len(drop_all) / self.client_num)]
                for s in range(0, len(drop_all),
                               round(len(drop_all) / self.client_num))
            ]

        graphs = []
        for owner in client_node_idx:
            nodes = client_node_idx[owner]
            sub_g = nx.Graph(nx.subgraph(G, nodes))
            if self.drop_edge:
                sub_g.remove_edges_from(
                    np.array(ovlap_graph.edges)[drop_client[owner]])
            graphs.append(from_networkx(sub_g))

        return graphs

    def __repr__(self):
        return f'{self.__class__.__name__}({self.client_num})'
