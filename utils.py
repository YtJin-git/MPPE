import os
import torch
import numpy as np
import random
import os
import yaml
import json

from tools.optimization import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

DIR_PATH = os.path.dirname(os.path.realpath(__file__))


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_args(filename, args):
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    for key, group in data_loaded.items():
        for key, val in group.items():
            setattr(args, key, val)


def write_json(filename, content):
    with open(filename, 'w') as f:
        json.dump(content, f)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def get_optimizer(model, config):
    if config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay, eps=1e-3)
    elif config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    return optimizer


def get_scheduler(optimizer, config, num_batches=-1):
    if not hasattr(config, 'scheduler'):
        return None
    if config.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)
    elif config.scheduler == 'linear_w_warmup' or config.scheduler == 'cosine_w_warmup':
        assert num_batches != -1
        num_training_steps = num_batches * config.epochs
        num_warmup_steps = int(config.warmup_proportion * num_training_steps)
        if config.scheduler == 'linear_w_warmup':
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
        if config.scheduler == 'cosine_w_warmup':
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    return scheduler


def step_scheduler(scheduler, config, bid, num_batches):
    if config.scheduler in ['StepLR']:
        if bid + 1 == num_batches:    # end of the epoch
            scheduler.step()
    elif config.scheduler in ['linear_w_warmup', 'cosine_w_warmup']:
        scheduler.step()

    return scheduler


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    # pdb.set_trace()
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def denseTosparse(mx, index):
    indices_row = index[0, :]
    indices_col = index[1, :]
    values = mx[indices_row, indices_col]
    del indices_row
    del indices_col
    return values


def sparseTodense(mx, values, index):
    # pdb.set_trace()
    zero_v = torch.zeros_like(mx)
    indices_row = index[0, :]
    indices_col = index[1, :]
    zero_v[indices_row, indices_col] = values
    # del zero_v
    del indices_row
    del indices_col
    # pdb.set_trace()
    return zero_v


def proximal_op(adj, estimated_adj, beta):
    index = adj.nonzero().t()
    zero_vec = torch.zeros_like(adj)
    Z = torch.where(adj == 0, zero_vec, estimated_adj)
    Z = torch.where(Z < 0, zero_vec, Z)

    Z_values = denseTosparse(Z, index)
    data = adj
    data_values = denseTosparse(data, index)
    for i in range(50):
        row_sum = torch.sum(sparseTodense(adj, data_values, index), 1) * beta
        data_values_addrowsum = row_sum[index[0, :][torch.arange(index.shape[1])]] + data_values[
            torch.arange(index.shape[1])]
        data_values = data_values * (Z_values / (data_values_addrowsum + 1e-8))
    # ----------------normalizition------------
    data_values = data_values / data_values.max()
    data = sparseTodense(adj, data_values, index)

    return data