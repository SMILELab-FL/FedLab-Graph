import torch
from fedlab.core.client.serial_trainer import SerialTrainer
from fedlab.utils.serialization import SerializationTool
from fedlab.utils.functional import AverageMeter


class NodeFullBatchSubsetSerialTrainer(SerialTrainer):
    def __init__(self,
                 model,
                 client_dict,
                 logger=None,
                 cuda=False,
                 args={
                     "epochs": 5,
                     "lr": 0.1,
                     "weight_decay": 5e-4
                 }) -> None:
        super(NodeFullBatchSubsetSerialTrainer, self).__init__(model=model,
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
            train_mask = data.train_mask
            output = self.model(data)
            loss = criterion(output[train_mask], data.y[train_mask])

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
            mask = data.val_mask if is_valid else data.test_mask

            output = self.model(data)
            labels = data.y[mask]
            loss = criterion(output[mask], labels)

            pred = output.argmax(dim=1)

            loss_.update(loss.item())
            acc_.update(torch.sum(pred[mask].eq(labels)).item(), int(mask.sum()))

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
            # print("client {} - val loss: {:.4f}, acc: {:.2f}".format(client_id, loss, acc))

        return loss_.avg, acc_.avg

