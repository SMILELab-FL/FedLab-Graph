# Codes below are copied and modified from
# https://github.com/alibaba/FederatedScope/blob/master/federatedscope/core/splitters/graph/graphtype_splitter.py
# Modified: using `fedlab.utils.dataset.functional.hetero_dir_partition` method

import numpy as np
from fedlab.utils.dataset.functional import hetero_dir_partition


class GraphTypeSplitter:
    def __init__(self, client_num, alpha=0.5):
        self.client_num = client_num
        self.alpha = alpha

    def __call__(self, dataset):
        r"""Split dataset via dirichlet distribution to generate non-i.i.d data split.

        Arguments:
            dataset (List or PyG.dataset): The datasets.

        Returns:
            data_list (List(List(PyG.data))): Splited dataset via dirichlet.
        """
        dataset = [ds for ds in dataset]
        label = np.array([ds.y.item() for ds in dataset])
        idx_slice = hetero_dir_partition(targets=label,
                                         num_clients=self.client_num,
                                         num_classes=len(np.unique(label)),
                                         dir_alpha=self.alpha)
        data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice.values()]
        return data_list

    def __repr__(self):
        return f'{self.__class__.__name__}()'
