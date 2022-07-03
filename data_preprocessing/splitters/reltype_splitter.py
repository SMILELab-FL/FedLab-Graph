# Codes below are copied and modified from
# https://github.com/alibaba/FederatedScope/blob/master/federatedscope/core/splitters/graph/reltype_splitter.py
# Modified: using `fedlab.utils.dataset.functional.hetero_dir_partition` method

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx, to_undirected
from torch_geometric.transforms import BaseTransform, RemoveIsolatedNodes

from fedlab.utils.dataset.functional import hetero_dir_partition


class RelTypeSplitter(BaseTransform):
    r"""
    Split Data into small data via dirichlet distribution to
    generate non-i.i.d data split.
    
    Arguments:
        client_num (int): Split data into client_num of pieces.
        alpha (float): parameter controlling the identicalness among clients.
        
    """

    def __init__(self, client_num, alpha=0.5, realloc_mask=False):
        self.client_num = client_num
        self.alpha = alpha
        self.realloc_mask = realloc_mask

    def __call__(self, data):
        data_list = []
        label = data.edge_type.numpy()  # only used for link prediction
        idx_slice = hetero_dir_partition(targets=label,
                                         num_clients=self.client_num,
                                         num_classes=len(np.unique(label)),
                                         dir_alpha=self.alpha)
        # Reallocation train/val/test mask
        train_ratio = data.train_edge_mask.sum().item() / data.num_edges
        valid_ratio = data.valid_edge_mask.sum().item() / data.num_edges
        test_ratio = data.test_edge_mask.sum().item() / data.num_edges
        for idx_j in idx_slice.values():
            edge_index = data.edge_index.T[idx_j].T
            edge_type = data.edge_type[idx_j]
            train_edge_mask = data.train_edge_mask[idx_j]
            valid_edge_mask = data.valid_edge_mask[idx_j]
            test_edge_mask = data.test_edge_mask[idx_j]
            if self.realloc_mask:
                num_edges = edge_index.size(-1)
                indices = torch.randperm(num_edges)
                train_edge_mask = torch.zeros(num_edges, dtype=torch.bool)
                train_edge_mask[indices[:round(train_ratio *
                                               num_edges)]] = True
                valid_edge_mask = torch.zeros(num_edges, dtype=torch.bool)
                valid_edge_mask[
                    indices[round(train_ratio *
                                  num_edges):-round(test_ratio *
                                                    num_edges)]] = True
                test_edge_mask = torch.zeros(num_edges, dtype=torch.bool)
                test_edge_mask[indices[-round(test_ratio * num_edges):]] = True
            sub_g = Data(x=data.x,
                         edge_index=edge_index,
                         index_orig=data.index_orig,
                         edge_type=edge_type,
                         train_edge_mask=train_edge_mask,
                         valid_edge_mask=valid_edge_mask,
                         test_edge_mask=test_edge_mask,
                         input_edge_index=to_undirected(
                             edge_index.T[train_edge_mask].T))
            data_list.append(sub_g)

        return data_list

    def __repr__(self):
        return f'{self.__class__.__name__}({self.client_num})'
