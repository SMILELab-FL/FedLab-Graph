import torch

from torch_geometric.data import Data
from torch_geometric import transforms
from torch_geometric.loader import GraphSAINTRandomWalkSampler, NeighborSampler
from fedlab.utils.dataset.partition import DataPartitioner
from data_preprocessing.splitters.splitter_builder import get_splitter


class LinkLevelPartitioner(DataPartitioner):
    """Graph data partitioner for link-level tasks.

        Partition link-task graph data given specific client number and splitter method.
        Currently, 2 supported partition schemes can be achieved by passing `splitter_type` parameter in initialization:

        - label_space_splitter:
          - ``split_type="rel_type"``: It is used to provide label distribution skew. For link-prediction classification
          tasks, e.g., relation prediction for knowledge graph completion, the existing triplets are split into the
          clients by latent dirichlet allocation (LDA)
            Optional params:
                - alpha (float): parameter controlling the identicalness in LDA among clients. Default to `0.5`
                - realloc_mask (boolean): controlling whether to re-allocation indices of clients' inner data. Default to `False`

        - meta_splitter:
        Simulate a real FL setting via splitting the graph based on the meta data or the values of those attributes, which
        naturally leads to non-iid
          - Applied dataset: RecSys, fixed 28 graphs for `ciao` and 3 graphs for `epinions`
          - There is no specific ``split_type`` parameter value setting for meta_splitter

        Args:
            data_name (str): Name of dataset, only ``"ciao"``, ``"epinions"``, ``"fb15k-237"``, ``"wn18"``, `
                `"fb15k"`` and ``"toy"`` are accepted currently.
            data_path (str): Path of dataset saving
            client_num (int): Number of clients for data partition.
            split_type (str): Split type, only ``"rel_type"`` and ``None`` are accepted by node tasks.
            transforms_funcs (dict): transforms functions for dataset, which are built from `get_transform` function.
            **kwargs: Params for split method.
        """

    def __init__(self,
                 data_name,
                 data_path,
                 client_num,
                 split_type,
                 loader_config={
                     'method': '',
                     'batch_size': 128,
                 },
                 transforms_funcs={},
                 **kwargs):
        self.data_name = data_name.lower()
        self.data_path = data_path
        self.splitter = get_splitter(split_type, client_num, **kwargs)
        self.loader_config = loader_config
        self.transforms_funcs = transforms_funcs
        self._perform_partition()

        self.client_num = client_num
        self.num_classes = len(self.global_dataset[0].edge_type.unique())
        self.num_features = self.global_dataset[0].x.shape[-1]

    def __getitem__(self, index):
        return self.data_local_dict[index]

    def __len__(self):
        return len(self.data_local_dict)

    def _perform_partition(self, tvt_num_split=[0.8, 0.1, 0.1]):
        if 'pre_transform' not in self.transforms_funcs:
            print(f'pre_transform is None! Using the default Constant pre_transform function')
            self.transforms_funcs['pre_transform'] = transforms.Constant(value=1.0,
                                                                         cat=False)
        if self.data_name in ['epinions', 'ciao']:
            from data_preprocessing.dataset.recom_sys import RecSys
            self.global_dataset = RecSys(self.data_path,
                                         self.data_name,
                                         FL=False,
                                         splits=tvt_num_split,
                                         **self.transforms_funcs)
            self.split_dataset = RecSys(self.data_path,
                                        self.data_name,
                                        FL=True,
                                        splits=tvt_num_split,
                                        **self.transforms_funcs)

        elif self.data_name in ['fb15k-237', 'wn18', 'fb15k', 'toy']:
            from data_preprocessing.dataset.kg import KG
            self.global_dataset = KG(self.data_path, self.data_name, **self.transforms_funcs)
            self.split_dataset = self.splitter(self.global_dataset[0])
        else:
            raise ValueError(f'No dataset named: {self.data_name}!')

        dataset = [ds for ds in self.split_dataset]
        self.data_local_dict = dict()

        for client_idx in range(len(dataset)):
            local_data = self._raw2loader(dataset[client_idx])
            self.data_local_dict[client_idx] = local_data

    def _raw2loader(self, raw_data):
        """Transform a graph into either dataloader for graph-sampling-based mini-batch training
        or still a graph for full-batch training.
        Arguments:
            raw_data (PyG.Data): a raw graph.
        :returns:
            sampler (object): a Dict containing loader and subgraph_sampler or still a PyG.Data object.
        """

        if self.loader_config['method'] == '':
            sampler = raw_data
        elif self.loader_config['method'] == 'graphsaint-rw':
            loader = GraphSAINTRandomWalkSampler(
                raw_data,
                batch_size=self.loader_config["batch_size"],
                walk_length=self.loader_config["walk_length"],
                num_steps=self.loader_config["num_steps"],
                sample_coverage=0)
            subgraph_sampler = NeighborSampler(raw_data.edge_index,
                                               sizes=[-1],
                                               batch_size=4096,
                                               shuffle=False,
                                               num_workers=self.loader_config["num_workers"])
            sampler = dict(data=raw_data,
                           train=loader,
                           val=subgraph_sampler,
                           test=subgraph_sampler)
        else:
            raise TypeError('Unsupported DataLoader Type {}'.format(
                self.loader_config['method']))

        return sampler
