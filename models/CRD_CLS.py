import os
from dgl.nn import GraphConv
import torch.nn as nn
import torch.nn.functional as F

os.environ["DGLBACKEND"] = "pytorch"

class CRD(nn.Module):
    def __init__(self, d_in, d_out, p):
        super(CRD, self).__init__()
        self.conv = GraphConv(d_in, d_out)
        self.p = p

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.p)
        return x

class CLS(torch.nn.Module):
    def __init__(self, d_in, d_out):
        super(CLS, self).__init__()
        self.conv = GraphConv(d_in, d_out)

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = F.log_softmax(x)
        return x

class Net(torch.nn.Module):
    def __init__(self, num_features, hidden, num_classes, dropout=0.2):
        super(Net, self).__init__()
        self.crd = CRD(num_features, hidden, dropout)
        self.cls = CLS(hidden, num_classes)

    def forward(self, g, in_feat):
        x = self.crd(g, in_feat)
        x = self.cls(g, x)
        return x
