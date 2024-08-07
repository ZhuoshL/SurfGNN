import torch
import os
import numpy as np
import torch.nn as nn
import math
import time
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
import scipy.io as sio
from torch_geometric.utils import dense_to_sparse, coalesce, dropout_adj
from net.inits import uniform

edge_file = "./edge"



def my_get_neighs_order(order_path):
    adj_mat_order = np.load(order_path)
    num_neigh = adj_mat_order.shape[1]
    neigh_orders = np.zeros((len(adj_mat_order), num_neigh + 1))
    neigh_orders[:, 1:num_neigh+1] = adj_mat_order
    neigh_orders[:, 0] = np.arange(len(adj_mat_order))
    neigh_order = np.ravel(neigh_orders).astype(np.int64)

    return neigh_order


def edge_build(batch_num, x, num_nodes):

    edge_index_file = {'81924': "edge_index_80k", '20484': "edge_index_20k", '5124': "edge_index_5k",
                       '1284': "edge_index_1k", '324': "edge_index_320"}
    edge_index = np.load(os.path.join(edge_file, edge_index_file[str(num_nodes)]) + '.npy')
    edge_index = edge_index.astype(np.int32)
    edge_att = np.ones(edge_index.shape[1])
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    edge_index = edge_index.long()
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes, num_nodes)
    all_edge_index, all_edge_att = [], []
    for i in range(batch_num):
        all_edge_index.append(edge_index + i * num_nodes)
        all_edge_att.append(edge_att)
    all_edge_index = np.hstack(all_edge_index)
    all_edge_att = np.hstack(all_edge_att)
    all_edge_index = torch.from_numpy(all_edge_index).long().to(x.device)
    all_edge_att = torch.from_numpy(all_edge_att.reshape(len(all_edge_att), 1)).float().to(x.device)

    return all_edge_index, all_edge_att


class Standard_GPool(nn.Module):
    def __init__(self, pooling_type='mean', ex_nodes=None):
        super(Standard_GPool, self).__init__()
        neigh_file = {'81924': "adj_edge_1ring_80k", '20484': "adj_edge_1ring_20k", '5124': "adj_edge_1ring_5k",
                      '1284': "adj_edge_1ring_1k", '324': "adj_edge_1ring_320"}

        self.neigh_orders = my_get_neighs_order(os.path.join(edge_file, neigh_file[str(ex_nodes)] + '.npy'))

        self.pooling_type = pooling_type
        self.num_neighbors = np.load(os.path.join(edge_file, neigh_file[str(ex_nodes)] + '.npy')).shape[1] + 1

    def forward(self, x, batch_num):
        x_batch = x.view(batch_num, -1, x.size(-1))
        batch_num, nodes_num, feature_num = x_batch.shape
        down_nodes_num = int((nodes_num + 12) / 4)
        x_add_zero = torch.zeros((batch_num, 1, x.size(-1)), device=x.device)
        x_batch = torch.cat([x_batch, x_add_zero], 1)

        gap_1 = int(down_nodes_num / 2)
        gap_2 = int(nodes_num / 2)
        gap_3 = int(nodes_num / 2 + down_nodes_num / 2)
        x1 = x_batch[:, self.neigh_orders[0: gap_1 * self.num_neighbors], :]
        x2 = x_batch[:, self.neigh_orders[gap_2 * self.num_neighbors:gap_3 * self.num_neighbors], :]
        x = torch.cat((x1, x2), dim=1).view(batch_num, down_nodes_num, self.num_neighbors, feature_num)

        if self.pooling_type == "mean":
            x = torch.mean(x, 2)
        if self.pooling_type == "max":
            x = torch.max(x, 2)

        assert (x.size() == torch.Size([batch_num, down_nodes_num, feature_num]))
        x = x.view(-1, feature_num)
        edge_index, edge_attr = edge_build(batch_num, x, num_nodes=down_nodes_num)
        return x, edge_index, edge_attr


class Global_GConv(nn.Module):
    def __init__(self, in_feats, out_feats, num_nodes):
        super(Global_GConv, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats
        self.num_nodes = num_nodes
        neigh_file = {'81924': "adj_edge_1ring_80k", '20484': "adj_edge_1ring_20k", '5124': "adj_edge_1ring_5k",
                      '1284': "adj_edge_1ring_1k", '324': "adj_edge_1ring_320"}

        self.neigh_orders = my_get_neighs_order(os.path.join(edge_file, neigh_file[str(self.num_nodes)] + '.npy'))

        self.edge_num = np.load(os.path.join(edge_file, neigh_file[str(self.num_nodes)] + '.npy')).shape[1] + 1
        self.weight = nn.Linear(self.edge_num * self.in_feats, self.out_feats)
        # self.reset_parameters(self.weight)

    def reset_parameters(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x, batch):
        neigh_orders = self.neigh_orders
        if batch > 1:
            for i in range(1, batch):
                neigh_orders = np.concatenate((neigh_orders, self.neigh_orders + i * self.num_nodes), 0)

        x_batch = x.view(batch, -1, x.size(-1))
        x_add_zero = torch.zeros((batch, 1, x.size(-1)), device=x_batch.device)
        x_batch = torch.concat([x_batch, x_add_zero], 1).view(-1, x.size(-1))
        mat = x_batch[neigh_orders, :].view(-1, self.edge_num * self.in_feats)
        out = self.weight(mat)

        return out
