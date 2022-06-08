import logging
logger = logging.getLogger(__name__)


def get_splitter(splitter_method, client_num, **args):
    # Delay import
    # generic splitter
    if splitter_method == 'lda':
        from data_preprocessing.splitters import LDASplitter
        splitter = LDASplitter(client_num, **args)
    # graph splitter
    elif splitter_method == 'louvain':
        from data_preprocessing.splitters import LouvainSplitter
        splitter = LouvainSplitter(client_num, **args)
    elif splitter_method == 'random':
        from data_preprocessing.splitters import RandomSplitter
        splitter = RandomSplitter(client_num, **args)
    elif splitter_method == 'rel_type':
        from data_preprocessing.splitters import RelTypeSplitter
        splitter = RelTypeSplitter(client_num, **args)
    elif splitter_method == 'graph_type':
        from data_preprocessing.splitters import GraphTypeSplitter
        splitter = GraphTypeSplitter(client_num, **args)
    elif splitter_method == 'scaffold':
        from data_preprocessing.splitters import ScaffoldSplitter
        splitter = ScaffoldSplitter(client_num, **args)
    elif splitter_method == 'scaffold_lda':
        from data_preprocessing.splitters import ScaffoldLdaSplitter
        splitter = ScaffoldLdaSplitter(client_num, **args)
    elif splitter_method == 'rand_chunk':
        from data_preprocessing.splitters import RandChunkSplitter
        splitter = RandChunkSplitter(client_num, **args)
    else:
        logger.warning(f'Splitter is none or not found.')
        splitter = None
    return splitter

