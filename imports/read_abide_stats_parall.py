'''
Author: Xiaoxiao Li
Date: 2019/02/24
'''

import os.path as osp
from os import listdir


import torch
import numpy as np

from torch_geometric.data import Data
import multiprocessing
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
from functools import partial
import deepdish as dd


edge_file = "./edge"
def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index -= node_slice[batch[row]].unsqueeze(0)

    slices = {'edge_index': edge_slice}
    if data.x is not None:
        slices['x'] = node_slice
    if data.edge_attr is not None:
        slices['edge_attr'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    if data.pos is not None:
        slices['pos'] = node_slice

    return data, slices


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1).squeeze() if len(seq) > 0 else None

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


def read_data(data_dir, num_nodes):
    onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
    onlyfiles.sort()
    batch = []
    pseudo = []
    y_list = []
    edge_att_list, edge_index_list, att_list = [], [], []

    # parallar computing
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    #pool =  MyPool(processes = cores)
    func = partial(read_single_data, data_dir, num_nodes=num_nodes)

    import timeit

    start = timeit.default_timer()

    res = pool.map(func, onlyfiles)

    pool.close()
    pool.join()

    stop = timeit.default_timer()

    print('Time: ', stop - start)



    for j in range(len(res)):
        edge_att_list.append(res[j][0])
        edge_index_list.append(res[j][1]+j*res[j][4])
        att_list.append(res[j][2])
        y_list.append(res[j][3])
        batch.append([j]*res[j][4])
        print(j)
        # pseudo.append(np.diag(np.ones(res[j][4])))
    print("OK")
    y_arr = np.stack(y_list)
    att_arr = np.concatenate(att_list, axis=0)
    edge_att_arr = np.hstack(edge_att_list)
    edge_index_arr = np.hstack(edge_index_list)

    print('ok')

    # pseudo_arr = np.tile(pseudo[0], (len(pseudo), 1))


    edge_att_torch = torch.from_numpy(edge_att_arr.reshape(len(edge_att_arr), 1)).float()
    att_torch = torch.from_numpy(att_arr).float()
    y_torch = torch.from_numpy(y_arr).float()  # classification
    batch_torch = torch.from_numpy(np.hstack(batch)).long()
    edge_index_torch = torch.from_numpy(edge_index_arr).long()
    print('ok')
    # pseudo_torch = torch.from_numpy(pseudo_arr).float()
    # data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch, pos=pseudo_torch)
    data = Data(x=att_torch, edge_index=edge_index_torch, y=y_torch, edge_attr=edge_att_torch)
    print('ok')
    data, slices = split(data, batch_torch)
    print('ok')
    return data, slices


def read_single_data(data_dir, filename, num_nodes):

    temp = dd.io.load(osp.join(data_dir, filename))

    edge_index_file = {324: "edge_index_320.npy", 1284: "edge_index_1k.npy", 5124: "edge_index_5k.npy",
                       20484: "edge_index_20k.npy", 81924: "edge_index_80k.npy"}
    edge_index = np.load(osp.join(edge_file, edge_index_file[num_nodes]))
    edge_index = edge_index.astype(np.int32)

    att = temp['feature'][()]
    att = np.concatenate((att[:int(num_nodes/2), :], att[40962:40962+int(num_nodes/2), :]), axis=0)
    label = temp['label'][()]
    edge_att = np.ones(edge_index.shape[1])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes, num_nodes)

    att_torch = torch.from_numpy(att).float()
    y_torch = torch.from_numpy(np.array(label)).float()  # classification

    data = Data(x=att_torch, edge_index=edge_index.long(), y=y_torch, edge_attr=edge_att)

    return edge_att.data.numpy(), edge_index.data.numpy(), att, label, num_nodes
