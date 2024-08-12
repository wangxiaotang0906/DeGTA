import inspect
import math
from typing import Any, Dict, Optional
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential
from torch_geometric.nn.conv import MessagePassing, GATConv
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_adj, to_dense_batch
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)



class WeightedGCNConv(MessagePassing):
    def __init__(self):
        super(WeightedGCNConv, self).__init__(aggr='mean')

    def forward(self, x, edge_index, edge_weight):
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_j, edge_weight):
        return edge_weight.unsqueeze(-1) * x_j


class DeGTAConv(torch.nn.Module):
    def __init__(
            self,
            ae_channels: int,
            pe_channels: int,
            se_channels: int,
            heads: int = 1,
            dropout: float = 0.0,
            act: str = 'relu',
            act_kwargs: Optional[Dict[str, Any]] = None,
            norm: Optional[str] = None,
            norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.ae_channels = ae_channels
        self.pe_channels = pe_channels
        self.se_channels = se_channels
        self.heads = heads
        self.dropout = dropout

        # multi-view encoder
        self.ae_encoder = Sequential(
            Linear(ae_channels, ae_channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(ae_channels * 2, ae_channels),
            Dropout(dropout),
        )

        self.pe_encoder = Sequential(
            Linear(pe_channels, pe_channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Linear(pe_channels * 2, pe_channels),
        )

        self.se_encoder = Sequential(
            Linear(se_channels, se_channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Linear(se_channels * 2, se_channels),
        )

        # local_channel
        self.ae_attn_l = GATConv(ae_channels, ae_channels)
        self.pe_attn_l = GATConv(pe_channels, pe_channels)
        self.se_attn_l = GATConv(se_channels, se_channels)
        self.weightedconv = WeightedGCNConv()

        # global_channel
        self.ae_attn_g = torch.nn.MultiheadAttention(ae_channels, heads)
        self.pe_attn_g = torch.nn.MultiheadAttention(pe_channels, heads)
        self.se_attn_g = torch.nn.MultiheadAttention(se_channels, heads)

        # multi-view adaptive integration
        self.a_l = torch.nn.Parameter(torch.tensor(0.33))
        self.b_l = torch.nn.Parameter(torch.tensor(0.33))
        self.c_l = torch.nn.Parameter(torch.tensor(0.33))
        self.a_g = torch.nn.Parameter(torch.tensor(0.33))
        self.b_g = torch.nn.Parameter(torch.tensor(0.33))
        self.c_g = torch.nn.Parameter(torch.tensor(0.33))

        # local-global adaptive integration
        self.localchannel = Linear(ae_channels, ae_channels, bias=False)
        self.globalchannel = Linear(ae_channels, ae_channels, bias=False)

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, ae_channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, ae_channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, ae_channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.ae_attn_l._reset_parameters()
        self.pe_attn_l._reset_parameters()
        self.se_attn_l._reset_parameters()
        self.ae_attn_g._reset_parameters()
        self.pe_attn_g._reset_parameters()
        self.se_attn_g._reset_parameters()
        self.localchannel._reset_parameters()
        self.globalchannel._reset_parameters()

        reset(self.ae_encoder)
        reset(self.pe_encoder)
        reset(self.se_encoder)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()

    def forward(
            self,
            ae: Tensor,
            pe: Tensor,
            se: Tensor,
            edge_index: Adj,
            K: int,
            batch: Optional[torch.Tensor] = None,
            **kwargs,
    ) -> Tensor:

        # multi_view encoder
        ae = self.ae_encoder(ae)
        pe = self.pe_encoder(pe)
        se = self.se_encoder(se)
        print("after encoder", ae, pe, se)

        # local_channel
        _, peattn = self.pe_attn_l(pe, edge_index, return_attention_weights=True)
        _, seattn = self.se_attn_l(se, edge_index, return_attention_weights=True)
        _, aeattn = self.ae_attn_l(ae, edge_index, return_attention_weights=True)
        local_attn = self.a_l * peattn[1] + self.b_l * seattn[1] + self.c_l * aeattn[1]
        out_local = self.weightedconv(ae, edge_index, local_attn)
        out_local = F.dropout(out_local, p=self.dropout, training=self.training)

        # global_channel
        _, peattn = self.pe_attn_g(pe, pe, pe, need_weights=True)
        _, seattn = self.se_attn_g(se, se, se, need_weights=True)
        _, aeattn = self.ae_attn_g(ae, ae, ae, need_weights=True)

        sample_attn = self.a_g * peattn + self.b_g * seattn
        adj = to_dense_adj(edge_index, max_num_nodes=ae.size(0))
        zero_vec = torch.zeros_like(adj)
        sample_attn = torch.where(adj > 0, zero_vec, sample_attn)

        # hard sampling
        values, indices = sample_attn.topk(K, dim=1, largest=True, sorted=True)
        mask = torch.zeros_like(sample_attn).scatter_(1, indices, torch.ones_like(values))
        sample_attn_masked = sample_attn * mask
        aeattn = aeattn * mask

        global_attn = (0.5 * self.a_g + 0.5 * self.b_g) * sample_attn_masked + self.c_g * aeattn
        column_means = torch.sum(global_attn, dim=1)
        global_attn = global_attn / column_means
        #         y_soft = sample_attn
        #         index = y_soft.max(dim=-1, keepdim=True)[1]
        #         y_hard = torch.zeros_like(sample_attn, memory_format=torch.legacy_contiguous_format).scatter_(-1, index, 1.0)
        #         ret = y_hard - y_soft.detach() + y_soft
        #         active_edge = ret[:,0]
        out_global = torch.matmul(global_attn, ae)
        out_global = F.dropout(out_global, p=self.dropout, training=self.training)

        # local_global integration
        out = self.localchannel(out_local) + self.globalchannel(out_global)

        if self.norm3 is not None:
            out = self.norm3(out)
        return out


