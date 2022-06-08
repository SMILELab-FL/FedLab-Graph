import numpy as np

from torch_geometric import transforms
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset, MoleculeNet
from data_preprocessing.splitters.splitter_builder import get_splitter
from fedlab.utils.dataset.partition import DataPartitioner


class GraphPartitionerGraphLevel(DataPartitioner):
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
        self.num_node_features = self.global_dataset.num_node_features

    def __getitem__(self, index):
        return self.data_local_dict[index]

    def __len__(self):
        return len(self.data_local_dict)

    def _get_numGraphLabels(self, dataset):
        s = set()
        for g in dataset:
            s.add(g.y.item())
        return len(s)

    def _perform_partition(self):
        transforms_funcs = dict()
        if self.data_name in [
            'MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'AIDS', 'NCI1',
            'ENZYMES', 'DD', 'PROTEINS', 'COLLAB', 'IMDB-BINARY', 'IMDB-MULTI',
            'REDDIT-BINARY'
        ]:
            # Add feat for datasets without attrubute
            if self.data_name in ['IMDB-BINARY', 'IMDB-MULTI'
                        ] and 'pre_transform' not in transforms_funcs:
                transforms_funcs['pre_transform'] = transforms.Constant(value=1.0,
                                                                        cat=False)
            dataset = TUDataset(self.data_path, self.data_name, **transforms_funcs)
            if self.splitter is None:
                raise ValueError('Please set the graph.')
            dataset = self.splitter(dataset)

        elif self.data_name in [
            'HIV', 'ESOL', 'FREESOLV', 'LIPO', 'PCBA', 'MUV', 'BACE', 'BBBP',
            'TOX21', 'TOXCAST', 'SIDER', 'CLINTOX'
        ]:
            dataset = MoleculeNet(self.data_path, self.data_name, **transforms_funcs)
            if self.splitter is None:
                raise ValueError('Please set the graph.')
            dataset = self.splitter(dataset)
        elif self.data_name.startswith('graph_multi_domain'.upper()):
            if self.data_name.endswith('mol'.upper()):
                dnames = ['MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'AIDS', 'NCI1']
            elif self.data_name.endswith('small'.upper()):
                dnames = [
                    'MUTAG', 'BZR', 'COX2', 'DHFR', 'PTC_MR', 'ENZYMES', 'DD',
                    'PROTEINS'
                ]
            elif self.data_name.endswith('mix'.upper()):
                if 'pre_transform' not in transforms_funcs:
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
            dataset = []
            # Some datasets contain x
            for dname in dnames:
                if dname.startswith('IMDB') or dname == 'COLLAB':
                    tmp_dataset = TUDataset(self.data_path, dname, **transforms_funcs)
                else:
                    tmp_dataset = TUDataset(
                        self.data_path,
                        dname,
                        pre_transform=None,
                        transform=transforms_funcs['transform']
                        if 'transform' in transforms_funcs else None)
                dataset.append(tmp_dataset)
        else:
            raise ValueError(f'No dataset named: {self.data_name}!')

        # get local dataset
        self.data_local_dict = dict()

        splits = [0.8, 0.1, 0.1]
        batch_size = 32

        # Build train/valid/test dataloader
        raw_train = []
        raw_valid = []
        raw_test = []
        for client_idx, gs in enumerate(dataset):
            index = np.random.permutation(np.arange(len(gs)))
            train_idx = index[:int(len(gs) * splits[0])]
            valid_idx = index[int(len(gs) *
                                  splits[0]):int(len(gs) * sum(splits[:2]))]
            test_idx = index[int(len(gs) * sum(splits[:2])):]
            dataloader = {
                'num_label': self._get_numGraphLabels(gs),
                'train': DataLoader([gs[idx] for idx in train_idx],
                                    batch_size,
                                    shuffle=True),
                'val': DataLoader([gs[idx] for idx in valid_idx],
                                  batch_size,
                                  shuffle=False),
                'test': DataLoader([gs[idx] for idx in test_idx],
                                   batch_size,
                                   shuffle=False),
            }
            self.data_local_dict[client_idx + 1] = dataloader
            raw_train = raw_train + [gs[idx] for idx in train_idx]
            raw_valid = raw_valid + [gs[idx] for idx in valid_idx]
            raw_test = raw_test + [gs[idx] for idx in test_idx]
        if not self.data_name.startswith('graph_multi_domain'.upper()):
            self.data_local_dict[0] = {
                'train': DataLoader(raw_train, batch_size, shuffle=True),
                'val': DataLoader(raw_valid, batch_size, shuffle=False),
                'test': DataLoader(raw_test, batch_size, shuffle=False),
            }
