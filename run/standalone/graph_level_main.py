import copy
import os
import argparse
import random
import sys
import torch
from pathlib import Path
import uuid

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))

from run.utils.functional import random_seed_init
from fedlab.utils.aggregator import Aggregators
from fedlab.utils.serialization import SerializationTool
from model.model_builder import get_gnn
from trainer.graph_serial_trainer import GraphMiniBatchSubsetSerialTrainer
from data_preprocessing.dataloader.dataloader_graph import GraphLevelPartitioner
from run.utils.transform_builder import get_transform

# configuration
parser = argparse.ArgumentParser(description="Standalone link training example")
parser.add_argument("--task", type=str, default='graph_level')
parser.add_argument("--dataset", type=str, default='hiv')
parser.add_argument("--data_root", type=str, default=BASE_DIR / 'data')
parser.add_argument("--total_client", type=int, default=20)
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
                           split_type='scaffold',
                           # loader_config=loader_config,
                           transforms_funcs=transforms_funcs,
                           batch_size=args.batch_size)

# get model
args.cuda = True if torch.cuda.is_available() else False
model = get_gnn(args, num_features=gs.num_features, num_classes=gs.num_classes)
temp_model = copy.deepcopy(model)

# FL settings
num_per_round = int(args.total_client * args.sample_ratio)
aggregator = Aggregators.fedavg_aggregate

# fedlab setup, load model in the best gpu
trainer = GraphMiniBatchSubsetSerialTrainer(model=model,
                                            client_dict=gs.data_local_dict,
                                            cuda=args.cuda,
                                            args={
                                                "epochs": args.epochs,
                                                "lr": args.lr,
                                                "weight_decay": args.weight_decay,
                                            })

print(trainer.gpu)

# train procedure
to_select = [i for i in range(total_client_num)]
acc_list = []
test_freq = 10

# best model pt
checkpt_folder = BASE_DIR / f'trained_model_dict/{args.task}/{args.dataset}/'
checkpt_folder.mkdir(parents=True, exist_ok=True)
checkpt_file = checkpt_folder / f'{uuid.uuid4().hex}.pt'
best_val_loss = 100

for rd in range(args.com_round):
    model_parameters = SerializationTool.serialize_model(model)
    # FL-train
    selection = random.sample(to_select, num_per_round)
    parameters_list = trainer.local_process(payload=[model_parameters],
                                            id_list=selection)
    SerializationTool.deserialize_model(model, aggregator(parameters_list))

    # valid evaluate
    val_loss, val_acc = trainer.evaluate(is_valid=True)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), checkpt_file)

    print("After round{} - val loss: {:.4f}, acc: {:.2f}".format(rd, val_loss, val_acc))

    # test evaluate
    if rd % test_freq == 0:
        temp_model.load_state_dict(torch.load(checkpt_file))
        temp_model_parameters = SerializationTool.serialize_model(temp_model)
        test_loss, test_acc = trainer.evaluate(temp_model_parameters, is_valid=False)
        print("After round{} - test loss: {:.4f}, acc: {:.2f}".format(rd, test_loss, test_acc))
