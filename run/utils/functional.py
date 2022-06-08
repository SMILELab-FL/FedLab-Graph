import torch
import numpy as np
import random
from fedlab.utils.functional import AverageMeter


def random_seed_init(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def evaluate(model, criterion, test_loader):
    """Evaluate classify task model accuracy."""
    model.eval()
    gpu = next(model.parameters()).device

    loss_ = AverageMeter()
    acc_ = AverageMeter()
    with torch.no_grad():
        for batch in test_loader:
            for data, edge_index, target in batch:
                data = data.cuda(gpu)
                edge_index = edge_index.cuda(gpu)
                target = target.cuda(gpu)

                output = model(data, edge_index)
                loss = criterion(output, target)

                _, predicted = torch.max(output, 1)
                loss_.update(loss.item())
                acc_.update(torch.sum(predicted.eq(target)).item(), len(target))

    return loss_.sum, acc_.avg
