import torch
from torch import nn
import torch.nn.functional as F

import os
import math


class SplitLayer(nn.Module):
    def __init__(self, input_dim, output_dim, m=1., omega=1.):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 4)
        self.input_dim=input_dim
        self.output_dim=output_dim
        self.m = m
        self.omega = omega

    def forward(self, x):
        o = self.linear(x)
        o = o.chunk(4, dim=-1)
        o = o[0].tanh() * o[1].sigmoid() * (o[2] * self.omega).sin() * (o[3] * self.omega).cos()
        return o * self.m

    def forward_with_activations(self, x):
        preact = self.linear(x)
        preact_tanh, preact_sigmoid, preact_sin, preact_cos = preact.chunk(4, dim=-1)
        act_tanh, act_sigmoid, act_sin, act_cos = preact_tanh.tanh(), preact_sigmoid.sigmoid(),(preact_sin* self.omega).sin(), (preact_cos* self.omega).cos()
        h = act_tanh * act_sigmoid * act_sin * act_cos
        
        h = h * self.m

        return h, [x, preact, preact_tanh, preact_sigmoid, preact_sin, preact_cos, act_tanh, act_sigmoid, act_sin, act_cos]


class SimpleSplitNet(nn.Module):
    def __init__(self, in_features, hidden_layers, out_features):
        super().__init__()        
        net = [SplitLayer(in_features, hidden_layers[0])]

        for fan_in, fan_out in zip(hidden_layers, hidden_layers[1:]):
            net.append(SplitLayer(fan_in, fan_out))

        net.append(nn.Linear(fan_out, out_features))
        self.net = nn.Sequential(*net)
        
    
    def forward(self, x):
        return self.net(x)


class ParallelSplitNet(nn.Module):
    def __init__(self, model_configs, out_features, encoding_size=128):
        super().__init__()
        
        
#         if not hasattr(m, '__len__'):
#             m = [m] * (hidden_layers+2)

        import rff
        self.encoding = rff.layers.GaussianEncoding(sigma=10.0, input_size=2, encoded_size=encoding_size)
        in_features = encoding_size * 2
        self.networks = nn.ModuleList([SimpleSplitNet(**k, in_features=in_features, out_features=out_features) for k in model_configs])

    def forward(self, x):
        x = self.encoding(x)        
        o = 0
        
        for net in self.networks:
            o = o + net(x)
        
        return o