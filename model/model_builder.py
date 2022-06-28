from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from model.gcn import GCN_Net
from model.sage import SAGE_Net
from model.gat import GAT_Net
from model.gin import GIN_Net
from model.gpr import GPR_Net
from model.link_level import GNN_Net_Link
from model.graph_level import GNN_Net_Graph
from model.mpnn import MPNNs2s


def get_gnn(args, num_features, num_classes, num_edge_features=0):
    if args.task == 'node_level':
        if args.model_type == 'gcn':
            # assume `data` is a dict where key is the client index, and value is a PyG object
            model = GCN_Net(num_features,
                            num_classes,
                            hidden=args.hidden,
                            max_depth=args.nlayers,
                            dropout=args.dropout)
        elif args.model_type == 'sage':
            model = SAGE_Net(num_features,
                             num_classes,
                             hidden=args.hidden,
                             max_depth=args.nlayers,
                             dropout=args.dropout)
        elif args.model_type == 'gat':
            model = GAT_Net(num_features,
                            num_classes,
                            hidden=args.hidden,
                            max_depth=args.nlayers,
                            dropout=args.dropout)
        elif args.model_type == 'gin':
            model = GIN_Net(num_features,
                            num_classes,
                            hidden=args.hidden,
                            max_depth=args.nlayers,
                            dropout=args.dropout)
        elif args.model_type == 'gpr':
            model = GPR_Net(num_features,
                            num_classes,
                            hidden=args.hidden,
                            K=args.nlayers,
                            dropout=args.dropout)
        else:
            raise ValueError('not recognized gnn model {}'.format(
                args.model_type))

    elif args.task == 'link_level':
        model = GNN_Net_Link(in_channels=num_features,
                             out_channels=num_classes,
                             hidden=args.hidden,
                             max_depth=args.nlayers,
                             dropout=args.dropout,
                             gnn=args.model_type)
    elif args.task == 'graph_level':
        if args.model_type == 'mpnn':
            model = MPNNs2s(in_channels=num_features,
                            out_channels=num_classes,
                            num_nn=num_edge_features,
                            hidden=args.hidden)
        else:
            model = GNN_Net_Graph(num_features,
                                  num_classes,
                                  hidden=args.hidden,
                                  max_depth=args.nlayers,
                                  dropout=args.dropout,
                                  gnn=args.model_type,
                                  pooling=args.graph_pooling)
    else:
        raise ValueError('not recognized data task {}'.format(
            args.task))
    return model
