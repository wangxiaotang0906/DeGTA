import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)
from layer import DeGTAConv

class DeGTA(torch.nn.Module):
    def __init__(self,
                 ae_channel: int,
                 K: int,
                 ae_dim: int,
                 pe_dim: int,
                 se_dim: int,
                 out_dim: int,
                 num_layers: int,
                 ):

        super().__init__()

        self.ae_emb = Linear(ae_channel, ae_dim)
        self.pe_emb = Linear(K, pe_dim)
        self.se_emb = Linear(K, se_dim)

        self.convs = ModuleList()
        for _ in range(num_layers):
            conv = DeGTAConv(ae_channels=ae_dim, pe_channels=pe_dim, se_channels=se_dim)
            self.convs.append(conv)
        self.classifier = Sequential(
            Linear(ae_dim, ae_dim // 2),
            ReLU(),
            Linear(ae_dim // 2, ae_dim // 4),
            ReLU(),
            Linear(ae_dim // 4, out_dim),
        )

    def forward(self, ae, pe, se, edge_index, K):
        ae = self.ae_emb(ae)
        pe = self.pe_emb(pe)
        se = self.se_emb(se)
        for conv in self.convs:
            ae = conv(ae, pe, se, edge_index=edge_index, K=K)
        x = self.classifier(ae)
        return x




