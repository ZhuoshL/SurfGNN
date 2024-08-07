import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import (add_self_loops, sort_edge_index, remove_self_loops)
from torch_sparse import spspmm
from torch.nn import Conv1d

from net.Nodal_selective_pooling import Nodal_GPool
from net.Score_weighted_fusion import TopKPooling
from net.Nodal_gronv import Nodal_GConv
from net.TSL import Global_GConv, Standard_GPool


class SurfGNN(nn.Module):

    def __init__(self, num_nodes, num_TSL, input_RSL, indim, ratio, nclass):
        super(SurfGNN, self).__init__()

        self.indim = indim
        self.dim0 = 1
        self.num_TSL = num_TSL
        self.num_RSL = 2
        dim_tsl_design = [8, 8, 16, 32]
        node_tsl_design = [81924, 20484, 5124, 1284, 324]
        self.dim_TSL = dim_tsl_design[-num_TSL:]
        self.dim_TSL.insert(0, self.dim0)
        self.nodes_TSL = node_tsl_design[
                         node_tsl_design.index(num_nodes):node_tsl_design.index(num_nodes) + num_TSL + 1]
        self.dim_RSL = [dim_tsl_design[-1], 128, 256]

        self.dim_fully = 256
        self.nclass = nclass
        self.input_RSL = input_RSL
        self.num_nodes = num_nodes

        self.tsl = nn.ModuleList()
        self.mid_k_rsl = 10
        self.mlp_gconv_rsl = []
        self.rsl = nn.ModuleList()
        self.res = nn.ModuleList()
        self.fully_feature = nn.ModuleList()


        for i in range(self.indim):
            tsl_list = []
            for tsl in range(self.num_TSL):
                tsl_list.append(Global_GConv(self.dim_TSL[tsl], self.dim_TSL[tsl + 1], num_nodes=self.nodes_TSL[tsl]))
                tsl_list.append(nn.LeakyReLU())
                tsl_list.append(
                    Global_GConv(self.dim_TSL[tsl + 1], self.dim_TSL[tsl + 1], num_nodes=self.nodes_TSL[tsl]))
                tsl_list.append(nn.LeakyReLU())
                tsl_list.append(Standard_GPool(ex_nodes=self.nodes_TSL[tsl]))
            mlp_gconv_list = []
            for rsl in range(self.num_RSL):
                mlp_gconv_list.append(nn.Sequential(nn.Linear(self.input_RSL, self.mid_k_rsl, bias=False), nn.ReLU(),
                                                    nn.Linear(self.mid_k_rsl,
                                                              self.dim_RSL[rsl + 1] * self.dim_RSL[rsl])))
                mlp_gconv_list.append(nn.Sequential(nn.Linear(self.input_RSL, self.mid_k_rsl, bias=False), nn.ReLU(),
                                                    nn.Linear(self.mid_k_rsl,
                                                              self.dim_RSL[rsl + 1] * self.dim_RSL[rsl + 1])))
            self.tsl.append(nn.ModuleList(tsl_list))
            self.mlp_gconv_rsl.append(nn.ModuleList(mlp_gconv_list))

            res_list = [Conv1d(self.dim_RSL[0], self.dim_RSL[2], 1),
                        Conv1d(self.dim_RSL[0], self.dim_RSL[1], 1),
                        Conv1d(self.dim_RSL[1], self.dim_RSL[2], 1)]
            fully_list = [torch.nn.Linear(self.dim_RSL[2] * 2, self.dim_fully),
                          torch.nn.Linear(self.dim_fully, self.nclass)]
            rsl_list = [Nodal_GConv(self.dim_RSL[0], self.dim_RSL[1], self.mlp_gconv_rsl[i][0], normalize=False),
                        nn.LeakyReLU(),
                        Nodal_GConv(self.dim_RSL[1], self.dim_RSL[1], self.mlp_gconv_rsl[i][1], normalize=False),
                        nn.LeakyReLU(),
                        Nodal_GPool(self.dim_RSL[1], ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid),

                        Nodal_GConv(self.dim_RSL[1], self.dim_RSL[2], self.mlp_gconv_rsl[i][2], normalize=False),
                        nn.LeakyReLU(),
                        Nodal_GConv(self.dim_RSL[2], self.dim_RSL[2], self.mlp_gconv_rsl[i][3], normalize=False),
                        nn.LeakyReLU(),
                        Nodal_GPool(self.dim_RSL[2], ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)]
            self.rsl.append(nn.ModuleList(rsl_list))
            self.res.append(nn.ModuleList(res_list))
            self.fully_feature.append(nn.ModuleList(fully_list))

        self.pool_score = TopKPooling(self.dim_RSL[2], ratio=1.0, multiplier=1, nonlinearity=torch.sigmoid)
        self.fully = nn.ModuleList([torch.nn.Linear(self.dim_RSL[2] * 2, self.dim_fully),
                                   torch.nn.Linear(self.dim_fully, self.nclass)])
        # for ff in range(self.indim):
        #     for rsl in range(self.num_RSL):
        #         self.init_parameters(self.mlp_gconv_rsl[ff][rsl])

    def init_parameters(self, model):
        for m in model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


    def augment_adj(self, edge_index, edge_weight, num_nodes, sparse=False):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight, num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight, num_nodes)
        if sparse:
            edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index, edge_weight,
                                             num_nodes, num_nodes, num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        return edge_index, edge_weight.unsqueeze(-1)

    def feature_connect(self, x, batch):
        x = x.chunk(self.indim, dim=0)
        batch = batch.chunk(self.indim, dim=0)
        out = []
        for i in range(len(x)):
            x_feature = x[i]
            batch_feature = batch[i]
            x_glopool = torch.cat([gmp(x_feature, batch_feature), gap(x_feature, batch_feature)], dim=1)
            x_out = F.relu(self.fully_feature[i][0](x_glopool))
            x_out = self.fully_feature[i][1](x_out)
            out.append(x_out)
        return torch.cat(out, dim=-1)

    def forward(self, x, batch, pos=None):
        batchsize = torch.max(batch) + 1
        input_feature = torch.split(x, 1, dim=1)
        batch = batch.view(batchsize, -1)
        batch = batch[:, 0:self.input_RSL].reshape(-1)
        batch_p = batch.unsqueeze(0).repeat_interleave(self.indim, dim=0)
        x_feature = []
        batch_together = []

        x_feature, batch_together, perm1_c, perm2_c = [[] for i in range(4)]

        for i in range(self.indim):
            x = input_feature[i]
            batch = batch_p[i]
            pos = torch.eye(self.input_RSL, device=x.device).repeat(batchsize, 1)

            for tsl in range(self.num_TSL):
                n_tsl = tsl * 5
                x = self.tsl[i][n_tsl](x, batchsize)
                x = self.tsl[i][n_tsl+1](x)
                x = self.tsl[i][n_tsl+2](x, batchsize)
                x = self.tsl[i][n_tsl+3](x)
                x, edge_index, edge_attr = self.tsl[i][n_tsl+4](x, batchsize)

            x_res0 = F.leaky_relu(self.res[i][0](x.reshape(batchsize, -1, x.size(-1)).permute(0, 2, 1)))
            x_res0 = x_res0.permute(0, 2, 1).reshape(x.size(0), -1)
            x_res1 = F.leaky_relu(self.res[i][1](x.reshape(batchsize, -1, x.size(-1)).permute(0, 2, 1)))

            x = self.rsl[i][0](x, edge_index, edge_attr, pos)
            x = self.rsl[i][1](x)
            x = self.rsl[i][2](x, edge_index, edge_attr, pos)
            x = self.rsl[i][3](x)

            x = x + x_res1.permute(0, 2, 1).reshape(-1, x.size(-1))
            x, edge_index, edge_attr, batch, perm1, _, _ = self.rsl[i][4](x, edge_index, edge_attr, batch)
            pos = pos[perm1]
            edge_index, edge_attr = self.augment_adj(edge_index, edge_attr.squeeze(), x.size(0))

            x_res2 = F.leaky_relu(self.res[i][2](x.reshape(batchsize, -1, x.size(-1)).permute(0, 2, 1)))

            x = self.rsl[i][5](x, edge_index, edge_attr, pos)
            x = self.rsl[i][6](x)
            x = self.rsl[i][7](x, edge_index, edge_attr, pos)
            x = self.rsl[i][8](x)

            x = x + x_res2.permute(0, 2, 1).reshape(-1, x.size(-1))
            x, edge_index, edge_attr, batch, perm2, _, _ = self.rsl[i][9](x, edge_index, edge_attr, batch)
            pos = pos[perm2]

            x = x + x_res0[perm1][perm2]

            x_feature.append(x)
            batch_together.append(batch)
            perm1_c.append(perm1.unsqueeze(0))
            perm2_c.append(perm2.unsqueeze(0))

        batch_together = torch.cat(batch_together, dim=0)
        x_feature = torch.cat(x_feature, dim=0)
        out_feature = self.feature_connect(x_feature, batch_together)

        [batch_all, index] = torch.sort(batch_together, stable=True)
        x_all = x_feature[index]
        x, batch, perm_all, _, score = self.pool_score(x_all, batch=batch_all)
        out = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        # out = torch.cat([gmp(x_all[perm_all], batch), gap(x_all[perm_all], batch)], dim=1)
        out = F.relu(self.fully[0](out))
        # out = F.dropout(out, p=0.3, training=self.training)
        out = self.fully[1](out)

        perm1_c = torch.cat(perm1_c, dim=0).permute(1, 0)
        perm2_c = torch.cat(perm2_c, dim=0).permute(1, 0)
        score = score.view(out.size(0), -1)

        return out, out_feature, self.pool_score.weight, score, perm1_c, perm2_c, perm_all

