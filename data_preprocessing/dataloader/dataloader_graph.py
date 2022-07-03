import numpy as np
from typing import Dict, Tuple, List
from torch_geometric import transforms
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset, MoleculeNet
from data_preprocessing.splitters.splitter_builder import get_splitter
from fedlab.utils.dataset.partition import DataPartitioner


class GraphLevelPartitioner(DataPartitioner):
    """Graph data partitioner for graph-level tasks.

           Partition graph-task graph data given specific client number and splitter method.
           Currently, 3 supported partition schemes can be achieved by passing `splitter_type` parameter in initialization:

           - label_space_splitter:
             - ``split_type="graph_type"``: It is used to provide label distribution skew.
             Split dataset via dirichlet distribution to generate non-i.i.d data split
               Optional params:
                   - alpha (float): parameter controlling the identicalness in LDA among clients. Default to `0.5`
                   - realloc_mask (boolean): controlling whether to re-allocation indices of clients' inner data. Default to `False`

           - instance_space_splitter:
             - ``split_type="scaffold"``: It is used to provide feature distribution skew (i.e., covariate shift).
           Implement it by sorting the graphs based on their values of a certain aspect, e.g., for Molhiv, molecules are
           sorted by their scaffold
             - Applied dataset with data.smiles property to do scaffold_split, e.g. `HIV`

           - multi_task_splitter:
           Designed for multi-task learning or personalized learning, where different clients have different tasks.

           Args:
               data_name (str): Name of dataset,
                Accepted:
                ['MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'AIDS', 'NCI1',
                'ENZYMES', 'DD', 'PROTEINS', 'COLLAB', 'IMDB-BINARY', 'IMDB-MULTI',
                'REDDIT-BINARY'], (using label_space_splitter)
                ['HIV', 'ESOL', 'FREESOLV', 'LIPO', 'PCBA', 'MUV', 'BACE', 'BBBP',
                'TOX21', 'TOXCAST', 'SIDER', 'CLINTOX'], (using instance_space_splitter)
                ['graph_multi_domain_mol', 'graph_multi_domain_small', 'graph_multi_domain_small'，
                'graph_multi_domain_mix', 'graph_multi_domain_biochem', 'graph_multi_domain_kddcupv1',
                'graph_multi_domain_kddcupv2'] (using multi_task_splitter, **WAITING TEST**)

               data_path (str): Path of dataset saving
               client_num (int): Number of clients for data partition.
               split_type (str): Split type, only ``"rel_type"`` and ``None`` are accepted by node tasks.
               transforms_funcs (dict): transforms functions for dataset, which are built from `get_transform` function.
               batch_size (int): Number of batch size for dataloader
               **kwargs: Params for split method.
           """

    def __init__(self,
                 data_name,
                 data_path,
                 client_num,
                 split_type,
                 transforms_funcs={},
                 batch_size=32,
                 **kwargs):
        self.data_name = data_name.upper()
        self.data_path = data_path
        self.splitter = get_splitter(split_type, client_num, **kwargs)
        self.transforms_funcs = transforms_funcs
        self.batch_size = batch_size
        self._perform_partition()

        self.client_num = client_num
        if not self.multi_graph:
            self.num_classes = self.global_dataset.num_classes  # self._get_numGraphLabels(self.global_dataset)
            self.num_features = self.global_dataset.num_features  # x.shape[-1]
            # self.num_edge_features

    def __getitem__(self, index):
        return self.data_local_dict[index]

    def __len__(self):
        return len(self.data_local_dict)

    def _get_numGraphLabels(self, dataset):
        s = set()
        for g in dataset:
            s.add(g.y.item())
        return len(s)

    def _perform_partition(self, tvt_num_split=[0.8, 0.1, 0.1]):
        """
        Get global data and perform data partition to get split data clients with self.splitter
        Codes are copied and modified from
        https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py

        Args:
            tvt_num_split (list): percentage list of train data, valid data and test data. Default=[0.8, 0.1, 0.1]
        """
        self.multi_graph = False

        if self.data_name in [
            'MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'AIDS', 'NCI1',
            'ENZYMES', 'DD', 'PROTEINS', 'COLLAB', 'IMDB-BINARY', 'IMDB-MULTI',
            'REDDIT-BINARY'
        ]:
            # Add feat for datasets without attrubute
            if self.data_name in ['IMDB-BINARY', 'IMDB-MULTI'
                                  ] and 'pre_transform' not in self.transforms_funcs:
                self.transforms_funcs['pre_transform'] = transforms.Constant(value=1.0,
                                                                             cat=False)
            self.global_dataset = TUDataset(self.data_path, self.data_name, **self.transforms_funcs)
            if self.splitter is None:
                raise ValueError('Please set the graph splitter.')
            self.split_dataset = self.splitter(self.global_dataset)

        elif self.data_name in [
            'HIV', 'ESOL', 'FREESOLV', 'LIPO', 'PCBA', 'MUV', 'BACE', 'BBBP',
            'TOX21', 'TOXCAST', 'SIDER', 'CLINTOX'
        ]:
            self.global_dataset = MoleculeNet(self.data_path, self.data_name, **self.transforms_funcs)
            if self.splitter is None:  # split_method: scaffold
                raise ValueError('Please set the graph splitter.')
            self.split_dataset = self.splitter(self.global_dataset)

        # --- multi_graph WAITING TEST ---
        elif self.data_name.startswith('graph_multi_domain'.upper()):
            self.multi_graph = True
            if self.data_name.endswith('mol'.upper()):
                dnames = ['MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'AIDS', 'NCI1']
            elif self.data_name.endswith('small'.upper()):
                dnames = [
                    'MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'ENZYMES', 'DD',
                    'PROTEINS'
                ]
            elif self.data_name.endswith('mix'.upper()):
                if 'pre_transform' not in self.transforms_funcs:
                    raise ValueError(f'pre_transform is None!')
                dnames = [
                    'MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'AIDS', 'NCI1',
                    'ENZYMES', 'DD', 'PROTEINS', 'COLLAB', 'IMDB-BINARY',
                    'IMDB-MULTI'
                ]
            elif self.data_name.endswith('biochem'.upper()):
                dnames = [
                    'MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'AIDS', 'NCI1',
                    'ENZYMES', 'DD', 'PROTEINS'
                ]
            # We provide kddcup dataset here.
            elif self.data_name.endswith('kddcupv1'.upper()):
                dnames = [
                    'MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'AIDS', 'NCI1',
                    'Mutagenicity', 'NCI109', 'PTC_MM', 'PTC_FR'
                ]
            elif self.data_name.endswith('kddcupv2'.upper()):
                dnames = ['TBD']
            else:
                raise ValueError(f'No dataset named: {self.data_name}!')
            self.global_dataset = []
            # Some datasets contain x
            for dname in dnames:
                if dname.startswith('IMDB') or dname == 'COLLAB':
                    tmp_dataset = TUDataset(self.data_path, dname, **self.transforms_funcs)
                else:
                    tmp_dataset = TUDataset(
                        self.data_path,
                        dname,
                        pre_transform=None,
                        transform=self.transforms_funcs['transform']
                        if 'transform' in self.transforms_funcs else None)
                self.global_dataset.append(tmp_dataset)
        else:
            raise ValueError(f'No dataset named: {self.data_name}!')

        # get local dataset
        self.data_local_dict = dict()
        self._raw2loader(self.split_dataset, tvt_num_split)

    def _raw2loader(self, raw_data_list, tvt_num_split):
        """
        Transform each split dataset into dataloader for graph-sampling-based mini-batch training
        Codes are copied from
        https://github.com/alibaba/FederatedScope/blob/master/federatedscope/gfl/dataloader/dataloader_graph.py

        Args:
            raw_data_list (List[PyG.Data]): list for raw pyg data.
        """
        # Build train/valid/test dataloader
        raw_train = []
        raw_valid = []
        raw_test = []
        for client_idx, gs in enumerate(raw_data_list):
            index = np.random.permutation(np.arange(len(gs)))
            train_idx = index[:int(len(gs) * tvt_num_split[0])]
            valid_idx = index[int(len(gs) *
                                  tvt_num_split[0]):int(len(gs) * sum(tvt_num_split[:2]))]
            test_idx = index[int(len(gs) * sum(tvt_num_split[:2])):]
            dataloader = {
                'num_label': self._get_numGraphLabels(gs),
                'train': DataLoader([gs[idx] for idx in train_idx],
                                    self.batch_size,
                                    shuffle=True),
                'val': DataLoader([gs[idx] for idx in valid_idx],
                                  self.batch_size,
                                  shuffle=False),
                'test': DataLoader([gs[idx] for idx in test_idx],
                                   self.batch_size,
                                   shuffle=False),
            }
            self.data_local_dict[client_idx] = dataloader
            # raw_train = raw_train + [gs[idx] for idx in train_idx]
            # raw_valid = raw_valid + [gs[idx] for idx in valid_idx]
            # raw_test = raw_test + [gs[idx] for idx in test_idx]
        # if not self.multi_graph:
        #     self.data_local_dict[self.client_num] = {
        #         'train': DataLoader(raw_train, self.batch_size, shuffle=True),
        #         'val': DataLoader(raw_valid, self.batch_size, shuffle=False),
        #         'test': DataLoader(raw_test, self.batch_size, shuffle=False),
        #     }
