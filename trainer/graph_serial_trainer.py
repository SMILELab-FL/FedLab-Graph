import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from fedlab.core.client.serial_trainer import SerialTrainer
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import AverageMeter


class GraphMiniBatchSubsetSerialTrainer(SerialTrainer):
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
        super(GraphMiniBatchSubsetSerialTrainer, self).__init__(model=model,
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
            for data in train_loader['train']:
                data = data.cuda(self.gpu) if self.cuda else data
                output = self.model(data)
                labels = data.y.squeeze(-1).long()
                if len(labels.size()) == 0:
                    labels = labels.unsqueeze(0)
                loss = criterion(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        # loss, acc = self._evaluate_alone(train_loader, is_valid=True)
        # self._LOGGER.info("LOCAL - val loss: {:.4f}, acc: {:.2f}".format(loss, acc))

        return self.model_parameters

    def _evaluate_alone(self, data_loader):
        """Evaluate classify task model accuracy."""
        self.model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        loss_ = AverageMeter()
        acc_ = AverageMeter()

        with torch.no_grad():
            for data in data_loader:
                data = data.cuda(self.gpu) if self.cuda else data
                output = self.model(data)
                labels = data.y.squeeze(-1).long()
                if len(labels.size()) == 0:
                    labels = labels.unsqueeze(0)
                loss = criterion(output, labels)

                pred = output.argmax(dim=1)
                loss_.update(loss.item())
                acc_.update(torch.sum(pred.eq(labels)).item(), len(labels))

        return loss_.avg, acc_.avg

    # evaluate all clients
    def evaluate(self, global_model_param, is_valid=True):
        loss_ = AverageMeter()
        acc_ = AverageMeter()

        for client_id in range(self.client_num):
            data_loader = self._get_dataloader(client_id)
            data_loader = data_loader['val'] if is_valid else data_loader['test']
            loss, acc = self._evaluate_alone(data_loader)
            loss_.update(loss)
            acc_.update(acc)
            print("client {} - val loss: {:.4f}, acc: {:.2f}".format(client_id, loss, acc))

        return loss_.avg, acc_.avg
