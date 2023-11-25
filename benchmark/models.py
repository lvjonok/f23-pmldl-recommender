"""models module contains implementations of the models classes for further benchmarking"""
# reference: https://medium.com/stanford-cs224w/recommender-systems-with-gnns-in-pyg-d8301178e377

# Standard library imports

# Third-party imports
import pandas as pd

pd.set_option("display.max_colwidth", None)

import torch
import torch.nn as nn
import torch.nn.functional as F

# from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import degree



class LightGCNConv(MessagePassing):
    def __init__(self, **kwargs):
        super().__init__(aggr="add")

    def forward(self, x, edge_index):
        # Compute normalization
        from_, to_ = edge_index
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

        # Start propagating messages (no update after aggregation)
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j


class NGCFConv(MessagePassing):
    def __init__(self, latent_dim, dropout, bias=True, **kwargs):
        super(NGCFConv, self).__init__(aggr="add", **kwargs)

        self.dropout = dropout

        self.lin_1 = nn.Linear(latent_dim, latent_dim, bias=bias)
        self.lin_2 = nn.Linear(latent_dim, latent_dim, bias=bias)

        self.init_parameters()

    def init_parameters(self):
        nn.init.xavier_uniform_(self.lin_1.weight)
        nn.init.xavier_uniform_(self.lin_2.weight)

    def forward(self, x, edge_index):
        # Compute normalization
        from_, to_ = edge_index
        deg = degree(to_, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[from_] * deg_inv_sqrt[to_]

        # Start propagating messages
        out = self.propagate(edge_index, x=(x, x), norm=norm)

        # Perform update after aggregation
        out += self.lin_1(x)
        out = F.dropout(out, self.dropout, self.training)
        return F.leaky_relu(out)

    def message(self, x_j, x_i, norm):
        return norm.view(-1, 1) * (self.lin_1(x_j) + self.lin_2(x_j * x_i))


class RecSysGNN(nn.Module):
    def __init__(
        self,
        latent_dim,
        num_layers,
        num_users,
        num_items,
        model,
        dropout=0.1,  # Only used in NGCF
    ):
        super(RecSysGNN, self).__init__()

        assert (
            model in ["NGCF", "LightGCN", "Custom"]
        ), f"model {model} not supported. Please choose from ['NGCF', 'LightGCN', 'Custom']"

        self.model = model
        self.embedding = nn.Embedding(num_users + num_items, latent_dim)

        if self.model == "NGCF":
            self.convs = nn.ModuleList(
                NGCFConv(latent_dim, dropout=dropout) for _ in range(num_layers)
            )
        elif self.model == "LightGCN":
            self.convs = nn.ModuleList(LightGCNConv() for _ in range(num_layers))

        self.init_parameters()

    def init_parameters(self):
        if self.model == "NGCF":
            nn.init.xavier_uniform_(self.embedding.weight, gain=1)
        elif self.model == "LightGCN":
            # Authors of LightGCN report higher results with normal initialization
            nn.init.normal_(self.embedding.weight, std=0.1)

    def forward(self, edge_index):
        emb0 = self.embedding.weight
        embs = [emb0]

        emb = emb0
        for conv in self.convs:
            emb = conv(x=emb, edge_index=edge_index)
            embs.append(emb)

        # TODO: adopt for custom model
        out = (
            torch.cat(embs, dim=-1)
            if self.model == "NGCF"
            else torch.mean(torch.stack(embs, dim=0), dim=0)
        )

        return emb0, out

    def encode_minibatch(self, users, pos_items, neg_items, edge_index):
        emb0, out = self(edge_index)
        return (
            out[users],
            out[pos_items],
            out[neg_items],
            emb0[users],
            emb0[pos_items],
            emb0[neg_items],
        )
