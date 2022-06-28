import numpy as np

from torch_geometric.datasets import Planetoid
from data_preprocessing.splitters.splitter_builder import get_splitter
from fedlab.utils.dataset.partition import DataPartitioner

INF = np.iinfo(np.int64).max


class NodeLevelPartitioner(DataPartitioner):
    """Graph data partitioner for node-level tasks.

        Partition node-task graph data given specific client number and splitter method.
        Currently, 3 supported partition schemes can be achieved by passing `splitter_type` parameter in initialization:

        - community_splitter:
        Simulate the locality-based standalone graph data, where nodes in the same client
        are densely connected while cross-client edges are unavailable.
          - ``split_type="louvain"``: use louvain community detection algorithm to partition a graph into several clusters,
          which are assigned to the clients, optionally with the objective of balancing the number of nodes in each client.
            Optional params:
                - delta: Control the gap between the number of nodes on the each client, default to 0.5)

        - random_splitter:
          - ``split_type="random"``: The node set of the original graph is randomly split into N subsets with or
          without intersections. And the subgraph of each client is deduced from the nodes assigned to that client.
            Optional params:
                - random_type (int): Type for random method. random_type (int): Random sampling method.
                    Accepted: 0 for ``"lognormal"`` and others for `"dirichlet"`
                - split_param (float): Param for the corresponding ``random_type`` sampling method.
                - sum_rate (float): Sum of samples of the unique nodes for each client, default to 1
                - overlapping_rate(float): Additional samples of overlapping data, eg. '0.4'
                - drop_edge(float): Drop edges (drop_edge / client_num) for each client whthin overlapping part.

        - meta_splitter:
        Simulate a real FL setting via splitting the graph based on the meta data or the values of those attributes, which
        naturally leads to non-iid
          - Applied dataset: FedDBLP, which is split by conference/organization
          - There is no specific ``split_type`` parameter value setting for meta_splitter

        Args:
            data_name (str): Name of dataset, only ``"cora"``, ``"citeseer"``, ``"pubmed"``, ``"dblp_conf"`` and `
                `"dblp_org"`` are accepted currently.
            data_path (str): Path of dataset saving
            client_num (int): Number of clients for data partition.
            split_type (str): Split type, only ``"louvain"``, ``"random"`` and ``None`` are accepted by node tasks.
            **kwargs: Params for split method.
        """

    def __init__(self,
                 data_name,
                 data_path,
                 client_num,
                 split_type,
                 **kwargs):
        self.data_name = data_name.lower()
        self.data_path = data_path
        self.splitter = get_splitter(split_type, client_num, **kwargs)
        self._perform_partition()

        self.client_num = client_num
        self.num_classes = self.global_dataset.num_classes
        self.num_features = self.global_dataset.num_features

    def __getitem__(self, index):
        return self.data_local_dict[index]

    def __len__(self):
        return len(self.data_local_dict)

    def _perform_partition(self, tvt_num_split=[0.5, 0.2, 0.3]):
        if self.data_name in ["cora", "citeseer", "pubmed"]:
            tvt_num_split = {
                'cora': [232, 542, INF],
                'citeseer': [332, 665, INF],
                'pubmed': [3943, 3943, INF],
            }

            self.global_dataset = Planetoid(self.data_path,
                                            self.data_name,
                                            split='random',
                                            num_train_per_class=tvt_num_split[self.data_name][0],
                                            num_val=tvt_num_split[self.data_name][1],
                                            num_test=tvt_num_split[self.data_name][2])
            self.split_dataset = self.splitter(self.global_dataset[0])
        elif self.data_name == "dblp_conf":
            from data_preprocessing.dataset.dblp_new import DBLPNew
            self.global_dataset = DBLPNew(self.data_path,
                                          FL=0,
                                          splits=tvt_num_split)
            self.split_dataset = DBLPNew(self.data_path,
                                         FL=1,
                                         splits=tvt_num_split)

        elif self.data_name == "dblp_org":
            from data_preprocessing.dataset.dblp_new import DBLPNew
            self.global_dataset = DBLPNew(self.data_path,
                                          FL=0,
                                          splits=tvt_num_split)
            self.split_dataset = DBLPNew(self.data_path,
                                         FL=2,
                                         splits=tvt_num_split)
        else:
            raise ValueError(f'No dataset named: {self.data_name}!')

        # get local dataset
        dataset = [ds for ds in self.split_dataset]

        self.data_local_dict = dict()

        for client_idx in range(len(dataset)):
            self.data_local_dict[client_idx] = dataset[client_idx]
