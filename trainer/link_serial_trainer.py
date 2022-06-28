import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from fedlab.core.client.serial_trainer import SerialTrainer
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import AverageMeter


MODE2MASK = {
    'train': 'train_edge_mask',
    'val': 'valid_edge_mask',
    'test': 'test_edge_mask'
}


class LinkFullBatchSubsetSerialTrainer(SerialTrainer):
    def __init__(self,
                 model,
                 client_dict,
                 logger=None,
                 cuda=False,
                 args={
                     "epochs": 5,
                     "lr": 0.1,
                     "weight_decay": 5e-4,
                 }) -> None:
        super(LinkFullBatchSubsetSerialTrainer, self).__init__(model=model,
                                                               client_num=len(client_dict),
                                                               cuda=cuda,
                                                               logger=logger)

        self.client_dict = client_dict
        self.args = args

    def _get_dataloader(self, client_id):
        return self.client_dict[client_id]

    def _train_alone(self, model_parameters, train_loader):
        epochs, lr, weight_decay = self.args["epochs"], self.args["lr"], self.args["weight_decay"]
        SerializationTool.deserialize_model(self._model, model_parameters)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self._model.parameters(), lr=lr, weight_decay=weight_decay)
        self._model.train()

        for _ in range(epochs):
            data = train_loader.cuda(self.gpu) if self.cuda else train_loader
            train_mask = data.train_edge_mask
            edges = data.edge_index.T[train_mask]

            h = self.model((data.x, data.edge_index))
            output = self.model.link_predictor(h, edges.T)
            labels = data.edge_type[train_mask]
            loss = criterion(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # loss, acc = self._evaluate_alone(train_loader, is_valid=True)
        # self._LOGGER.info("LOCAL - val loss: {:.4f}, acc: {:.2f}".format(loss, acc))

        return self.model_parameters

    def _evaluate_alone(self, data_loader, is_valid=True):
        """Evaluate classify task model accuracy."""
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        loss_ = AverageMeter()
        acc_ = AverageMeter()

        with torch.no_grad():
            data = data_loader.cuda(self.gpu) if self.cuda else data_loader
            mask = data.valid_edge_mask if is_valid else data.test_edge_mask

            edges = data.edge_index.T[mask]
            h = self.model((data.x, data.edge_index))
            output = self.model.link_predictor(h, edges.T)
            labels = data.edge_type[mask]
            loss = criterion(output, labels)

            pred = output.argmax(dim=1)

            loss_.update(loss.item())
            acc_.update(torch.sum(pred.eq(labels)).item(), int(mask.sum()))

        return loss_.avg, acc_.avg

    # evaluate all clients
    def evaluate(self, global_model_param, is_valid=True):
        loss_ = AverageMeter()
        acc_ = AverageMeter()

        for client_id in range(self.client_num):
            data_loader = self._get_dataloader(client_id)
            loss, acc = self._evaluate_alone(data_loader, is_valid)
            loss_.update(loss)
            acc_.update(acc)
            print("client {} - val loss: {:.4f}, acc: {:.2f}".format(client_id, loss, acc))

        return loss_.avg, acc_.avg
