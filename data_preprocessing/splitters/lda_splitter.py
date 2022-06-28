import numpy as np
from fedlab.utils.dataset.functional import hetero_dir_partition


class LDASplitter(object):
    def __init__(self, client_num, alpha=0.5):
        self.client_num = client_num
        self.alpha = alpha

    def __call__(self, dataset):
        dataset = [ds for ds in dataset]
        label = np.array([y for x, y in dataset])
        idx_slice = hetero_dir_partition(targets=label,
                                         num_clients=self.client_num,
                                         num_classes=len(np.unique(label)),
                                         dir_alpha=self.alpha)
        data_list = [[dataset[idx] for idx in idxs] for idxs in idx_slice.values()]
        return data_list

    def __repr__(self):
        return f'{self.__class__.__name__}(client_num={self.client_num}, alpha={self.alpha})'
