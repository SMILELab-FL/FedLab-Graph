# Codes below are copied from
# https://github.com/alibaba/FederatedScope/blob/master/federatedscope/core/auxiliaries/splitter_builder.py

import logging
logger = logging.getLogger(__name__)


def get_splitter(splitter_method, client_num, **kwargs):
    # Delay import
    # graph splitter
    if splitter_method == 'louvain':
        from data_preprocessing.splitters.louvain_splitter import LouvainSplitter
        splitter = LouvainSplitter(client_num, **kwargs)
    elif splitter_method == 'random':
        from data_preprocessing.splitters.random_splitter import RandomSplitter
        splitter = RandomSplitter(client_num, **kwargs)
    elif splitter_method == 'rel_type':
        from data_preprocessing.splitters.reltype_splitter import RelTypeSplitter
        splitter = RelTypeSplitter(client_num, **kwargs)
    elif splitter_method == 'graph_type':
        from data_preprocessing.splitters.graphtype_splitter import GraphTypeSplitter
        splitter = GraphTypeSplitter(client_num, **kwargs)
    elif splitter_method == 'scaffold':
        from data_preprocessing.splitters.scaffold_splitter import ScaffoldSplitter
        splitter = ScaffoldSplitter(client_num, **kwargs)
    elif splitter_method == 'scaffold_lda':
        from data_preprocessing.splitters.scaffold_lda_splitter import ScaffoldLdaSplitter
        splitter = ScaffoldLdaSplitter(client_num, **kwargs)
    elif splitter_method == 'rand_chunk':
        from data_preprocessing.splitters.randchunk_splitter import RandChunkSplitter
        splitter = RandChunkSplitter(client_num, **kwargs)
    else:
        logger.warning(f'Splitter is none or not found.')
        splitter = None
    return splitter

