from torch_geometric.nn.conv import MessagePassing
from typing import Union, Optional, Callable
import torch
from torch.nn import Parameter
from torch_scatter import scatter_add, scatter_max
from torch_geometric.utils import softmax, add_remaining_self_loops
from torch_geometric.nn.inits import uniform
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.nn import GCNConv


class NodeInformationScore(MessagePassing):
    def __init__(self, improved=False, cached=False, **kwargs):
        super(NodeInformationScore, self).__init__(aggr='add', **kwargs)

        self.improved = improved
        self.cached = cached
        self.cached_result = None
        self.cached_num_edges = None

    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),), dtype=dtype, device=edge_index.device)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        edge_index, edge_weight = add_remaining_self_loops(edge_index, edge_weight, 0, num_nodes)

        row, col = edge_index
        expand_deg = torch.zeros((edge_weight.size(0),), dtype=dtype, device=edge_index.device)
        expand_deg[-num_nodes:] = torch.ones((num_nodes,), dtype=dtype, device=edge_index.device)
        a = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        b = expand_deg - a.squeeze(-1)

        return edge_index, b

    def forward(self, x, edge_index, edge_weight):
        if self.cached and self.cached_result is not None:
            if edge_index.size(1) != self.cached_num_edges:
                raise RuntimeError(
                    'Cached {} number of edges, but found {}'.format(self.cached_num_edges, edge_index.size(1)))

        if not self.cached or self.cached_result is None:
            self.cached_num_edges = edge_index.size(1)
            edge_index, norm = self.norm(edge_index, x.size(0), edge_weight, x.dtype)
            self.cached_result = edge_index, norm

        edge_index, norm = self.cached_result

        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        return aggr_out


def topk(x, ratio, batch, min_score=None, tol=1e-7):
    if min_score is not None:
        scores_max = scatter_max(x, batch)[0][batch] - tol
        scores_min = scores_max.clamp(max=min_score)

        perm = (x > scores_min).nonzero(as_tuple=False).view(-1)
    else:
        num_nodes = scatter_add(batch.new_ones(x.size(0)), batch, dim=0)
        batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()

        cum_num_nodes = torch.cat(
            [num_nodes.new_zeros(1),
             num_nodes.cumsum(dim=0)[:-1]], dim=0)

        index = torch.arange(batch.size(0), dtype=torch.long, device=x.device)
        index = (index - cum_num_nodes[batch]) + (batch * max_num_nodes)

        dense_x = x.new_full((batch_size * max_num_nodes,),
                             torch.finfo(x.dtype).min)
        dense_x[index] = x
        dense_x = dense_x.view(batch_size, max_num_nodes)

        _, perm = dense_x.sort(dim=-1, descending=True)

        perm = perm + cum_num_nodes.view(-1, 1)
        perm = perm.view(-1)

        if isinstance(ratio, int):
            k = num_nodes.new_full((num_nodes.size(0),), ratio)
            k = torch.min(k, num_nodes)
        else:
            k = (ratio * num_nodes.to(torch.float)).ceil().to(torch.long)

        mask = [
            torch.arange(k[i], dtype=torch.long, device=x.device) +
            i * max_num_nodes for i in range(batch_size)
        ]
        mask = torch.cat(mask, dim=0)

        perm = perm[mask]

    return perm


def filter_adj(edge_index, edge_attr, perm, num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    mask = perm.new_full((num_nodes,), -1)
    i = torch.arange(perm.size(0), dtype=torch.long, device=perm.device)
    mask[perm] = i

    row, col = edge_index
    row, col = mask[row], mask[col]
    mask = (row >= 0) & (col >= 0)
    row, col = row[mask], col[mask]

    if edge_attr is not None:
        edge_attr = edge_attr[mask]

    return torch.stack([row, col], dim=0), edge_attr


class Nodal_GPool(torch.nn.Module):
    def __init__(self, in_channels: int, ratio: Union[int, float] = 0.5,
                 min_score: Optional[float] = None, multiplier: float = 1.,
                 nonlinearity: Callable = torch.tanh):
        super(Nodal_GPool, self).__init__()

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier
        self.nonlinearity = nonlinearity
        self.calc_information_score = NodeInformationScore()


    def forward(self, x, edge_index=None, edge_attr=None, batch=None, attn=None, weight=None):


        attn = x if attn is None else attn
        attn = attn.unsqueeze(-1) if attn.dim() == 1 else attn


        x_information_score = self.calc_information_score(x, edge_index, edge_attr)
        score_h = torch.sum(torch.abs(x_information_score), dim=1)


        if self.min_score is None:
            # score = 0.5 * self.nonlinearity(score_w / self.weight.norm(p=2, dim=-1)) + score_h * 0.5
            score = self.nonlinearity(score_h)
        else:
            score = softmax(score_h, batch)
        perm = topk(abs(score), self.ratio, batch, self.min_score)
        x = x[perm]  # * score[perm].view(-1, 1)
        x = self.multiplier * x if self.multiplier != 1 else x
        batch = batch[perm]
        if edge_attr is None:
            return x, batch, perm, score, score[perm]
        else:
            edge_index, edge_attr = filter_adj(edge_index, edge_attr, perm, num_nodes=score.size(0))
            return x, edge_index, edge_attr, batch, perm, score, score[perm]

    def __repr__(self):
        return '{}({}, {}={}, multiplier={})'.format(
            self.__class__.__name__, self.in_channels,
            'ratio' if self.min_score is None else 'min_score',
            self.ratio if self.min_score is None else self.min_score,
            self.multiplier)

#
# # # The topkpool_my is almost the same with torch_geometric.nn.TopKPooling
# # # Actually we just change the output of the class TopKPooling
# # # We add the score which we need to use when calling it
