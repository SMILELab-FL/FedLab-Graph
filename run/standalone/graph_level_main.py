import copy
import os
import argparse
import random
import sys
import torch
from pathlib import Path
import uuid

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append("../../")

from run.utils.functional import random_seed_init
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import get_best_gpu
from model.model_builder import get_gnn
from trainer.graph_serial_trainer import GraphMiniBatchSubsetSerialTrainer
from data_preprocessing.dataloader.dataloader_graph import GraphLevelPartitioner
from run.utils.transform_builder import get_transform

# configuration
parser = argparse.ArgumentParser(description="Standalone link training example")
parser.add_argument("--task", type=str, default='graph_level')
parser.add_argument("--dataset", type=str, default='IMDB-BINARY')
parser.add_argument("--data_root", type=str, default='../../data/')
parser.add_argument("--total_client", type=int, default=10)
parser.add_argument("--com_round", type=int, default=50)
parser.add_argument("--sample_ratio", type=float, default=1)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--lr", type=float, default=0.02)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--batch_size', type=int, default=32)


# model setting
parser.add_argument("--model_type", type=str, default='gcn')
parser.add_argument('--hidden', type=int, default=64)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('--graph_pooling', type=str, default='add')


args = parser.parse_args()
random_seed_init(42)

# get dataset
total_client_num = args.total_client
# loader config for
loader_config = {
    'method': ''
}

# necessary for link-level dataset to generate `data.x`
transforms_funcs = get_transform({'pre_transform': ['Constant', {'value': 1.0, 'cat': False}]},
                                 'torch_geometric')
gs = GraphLevelPartitioner(data_name=args.dataset,
                           data_path=Path(args.data_root) / args.task,
                           client_num=total_client_num,
                           split_type='graph_type',
                           # loader_config=loader_config,
                           transforms_funcs=transforms_funcs,
                           batch_size=args.batch_size)

# get model
args.cuda = True if torch.cuda.is_available() else False
device = get_best_gpu() if args.cuda else 'cpu'
model = get_gnn(args, num_features=gs.num_features, num_classes=gs.num_classes).to(device)

# FL settings
num_per_round = int(args.total_client * args.sample_ratio)
aggregator = Aggregators.fedavg_aggregate

# fedlab setup
trainer = GraphMiniBatchSubsetSerialTrainer(model=model,
                                            client_dict=gs.data_local_dict,
                                            cuda=args.cuda,
                                            args={
                                                "epochs": args.epochs,
                                                "lr": args.lr,
                                                "weight_decay": args.weight_decay,
                                            })

# train procedure
to_select = [i for i in range(total_client_num)]
acc_list = []
test_pre_round = 10

for rd in range(args.com_round):
    model_parameters = SerializationTool.serialize_model(model)
    # valid evaluate
    loss, acc = trainer.evaluate(model_parameters, is_valid=True)
    print("Before round{} - val loss: {:.4f}, acc: {:.2f}".format(rd, loss, acc))

    # test evaluate
    if rd % test_pre_round == 0 and rd != 0:
        loss, acc = trainer.evaluate(model_parameters, is_valid=False)
        acc_list.append(acc)
        print("Round {} - test loss: {:.4f}, acc: {:.2f}".format(rd, loss, acc))

    # FL-train
    selection = random.sample(to_select, num_per_round)
    parameters_list = trainer.local_process(payload=[model_parameters],
                                            id_list=selection)
    SerializationTool.deserialize_model(model, aggregator(parameters_list))
