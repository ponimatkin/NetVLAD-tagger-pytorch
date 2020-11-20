import torch.nn as nn
from model.netvlad import NetVLAD


class NetVLADTagger(nn.Module):
    def __init__(self, cluster_size, feature_size, n_units):
        super(NetVLADTagger, self).__init__()
        self.vlad = NetVLAD(cluster_size=cluster_size, feature_size=feature_size)
        units = list()
        for unit in range(n_units):
            units.append(ResidualUnit(cluster_size * feature_size))
            units.append(nn.Dropout(p=0.5))

        self.units = nn.Sequential(*units)
        self.output = nn.Linear(cluster_size * feature_size, 2)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.vlad(x)
        x = self.units(x)
        x = self.output(x)

        return x

class ResidualUnit(nn.Module):

    def __init__(self, dim):
        super(ResidualUnit, self).__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.activation = nn.CELU(inplace=True)

    def forward(self, x):
        identity = x

        x = self.linear1(x)
        x = self.activation(x)
        x = self.bn1(x)

        x = self.linear2(x)

        x = x + identity
        x = self.activation(x)
        x = self.bn2(x)

        return x