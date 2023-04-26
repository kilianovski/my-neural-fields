import torch
from torch import nn
import torch.nn.functional as F

import os
import math


class SplitLayer(nn.Module):
    def __init__(self, input_dim, output_dim, m=1.):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 4)
        self.m = m

    def forward(self, x):
        o = self.linear(x)
        o = o.chunk(4, dim=-1)
        o = o[0].tanh() * o[1].sigmoid() * o[2].sin() * o[3].cos()
        return o * self.m

    def forward_with_activations(self, x):
        preact = self.linear(x)
        preact_tanh, preact_sigmoid, preact_sin, preact_cos = preact.chunk(4, dim=-1)
        act_tanh, act_sigmoid, act_sin, act_cos = preact_tanh.tanh(), preact_sigmoid.sigmoid(), preact_sin.sin(), preact_cos.cos()
        h = act_tanh * act_sigmoid * act_sin * act_cos

        return h, [x, preact, preact_tanh, preact_sigmoid, preact_sin, preact_cos, act_tanh, act_sigmoid, act_sin, act_cos]


class SplitLayerLin(nn.Module):
    def __init__(self, input_dim, output_dim, m=1.):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 4)
        self.m = m

    def forward(self, x):
        o = self.linear(x)
        o = o.chunk(4, dim=-1)
        o = o[0].tanh() * o[1].sigmoid() * o[2].sin() * o[3]
        return o * self.m

    # def forward_with_activations(self, x):
    #     preact = self.linear(x)
    #     preact_tanh, preact_sigmoid, preact_sin, preact_cos = preact.chunk(4, dim=-1)
    #     act_tanh, act_sigmoid, act_sin, act_cos = preact_tanh.tanh(), preact_sigmoid.sigmoid(), preact_sin.sin(), preact_cos.cos()
    #     h = act_tanh * act_sigmoid * act_sin * act_cos

    #     return h, [x, preact, preact_tanh, preact_sigmoid, preact_sin, preact_cos, act_tanh, act_sigmoid, act_sin, act_cos]

class SplitNetLinPosEnc(nn.Module):
    def __init__(self, 
                 in_features=2,
                 encoding_size=64,
                 hidden_features=64, 
                 hidden_layers=1, 
                 out_features=1, 
                 outermost_linear=True,
                 m=1.
                 ):
        super().__init__()
        import rff
        self.encoding = rff.layers.GaussianEncoding(sigma=10.0, input_size=in_features, encoded_size=encoding_size)
        self.net = []
        layer_cls = SplitLayerLin
        self.net += [layer_cls(encoding_size*2, hidden_features, m=m)]
        self.net += [layer_cls(hidden_features, hidden_features, m=m) for _ in range(hidden_layers)]

        if outermost_linear:
            self.net += [nn.Linear(hidden_features, out_features)]
        else:
            self.net += [layer_cls(hidden_features, out_features, m=m)]

        self.net = nn.Sequential(*self.net)
        self.outermost_linear = outermost_linear

    def forward(self, x):
        x = self.encoding(x)
        return self.net(x)


class SplitNet(nn.Module):
    def __init__(self, 
                 in_features=2, 
                 hidden_features=64, 
                 hidden_layers=1, 
                 out_features=1, 
                 outermost_linear=True,
                 m=1.
                 ):
        super().__init__()
        self.net = []

        self.net += [SplitLayer(in_features, hidden_features, m=m)]
        self.net += [SplitLayer(hidden_features, hidden_features, m=m) for _ in range(hidden_layers)]

        if outermost_linear:
            self.net += [nn.Linear(hidden_features, out_features)]
        else:
            self.net += [SplitLayer(hidden_features, out_features, m=m)]

        self.net = nn.Sequential(*self.net)
        self.outermost_linear = outermost_linear

    def forward(self, x):
        return self.net(x)

    def forward_with_activations(self, x):
        h = x
        intermediate_acts = []

        for layer in self.net:
            if isinstance(layer, SplitLayer):
                h, acts = layer.forward_with_activations(h)
            else:
                h = layer(h)
                acts = []

            intermediate_acts.append((h, acts))

        return h, intermediate_acts


class SplitNetPosEnc(nn.Module):
    def __init__(self, 
                 in_features=2,
                 encoding_size=64,
                 hidden_features=64, 
                 hidden_layers=1, 
                 out_features=1, 
                 outermost_linear=True,
                 m=1.
                 ):
        super().__init__()
        import rff
        self.encoding = rff.layers.GaussianEncoding(sigma=10.0, input_size=in_features, encoded_size=encoding_size)
        self.net = []

        self.net += [SplitLayer(encoding_size*2, hidden_features, m=m)]
        self.net += [SplitLayer(hidden_features, hidden_features, m=m) for _ in range(hidden_layers)]

        if outermost_linear:
            self.net += [nn.Linear(hidden_features, out_features)]
        else:
            self.net += [SplitLayer(hidden_features, out_features, m=m)]

        self.net = nn.Sequential(*self.net)
        self.outermost_linear = outermost_linear

    def forward(self, x):
        x = self.encoding(x)
        return self.net(x)

    def forward_with_activations(self, x):
        x = self.encoding(x)
        h = x
        intermediate_acts = []

        for layer in self.net:
            if isinstance(layer, SplitLayer):
                h, acts = layer.forward_with_activations(h)
            else:
                h = layer(h)
                acts = []

            intermediate_acts.append((h, acts))

        return h, intermediate_acts

import plotly.express as px
import pandas as pd
import numpy as np

import lovely_tensors as lt

lt.monkey_patch()


def to_np(t):
    a = t.detach().flatten().cpu().numpy()
    a = np.random.choice(a, size=1_000, replace=False)
    return a


def U(*shape):
    x = torch.rand(*shape)
    x = (x - 0.5) * 2.0
    return x


def plot_acts(acts, title="acts"):
    x, preact, preact_tanh, preact_sigmoid, preact_sin, preact_cos, act_tanh, act_sigmoid, act_sin, act_cos = acts

    preact_tanh = to_np(preact_tanh)
    preact_sigmoid = to_np(preact_sigmoid)
    preact_sin = to_np(preact_sin)
    preact_cos = to_np(preact_cos)
    act_tanh = to_np(act_tanh)
    act_sigmoid = to_np(act_sigmoid)
    act_sin = to_np(act_sin)
    act_cos = to_np(act_cos)

    # Create a dataframe from your arrays
    data = {
        # "preact_tanh": preact_tanh,
        # "preact_sigmoid": preact_sigmoid,
        # "preact_sin": preact_sin,
        # "preact_cos": preact_cos,
        "act_tanh": act_tanh,
        "act_sigmoid": act_sigmoid,
        "act_sin": act_sin,
        "act_cos": act_cos,
    }

    df = pd.DataFrame(data)

    # Melt the dataframe to a long format for easier plotting
    df_melted = df.melt(var_name="activation_type", value_name="value")

    # Plot histograms using Plotly Express
    fig = px.histogram(
        df_melted,
        x="value",
        color="activation_type",
        facet_col="activation_type",
        facet_col_wrap=4,
        histnorm="probability density",
        title=title,
    )

    fig.update_layout(
        autosize=False,
        width=1000,
        height=600,
    )

    fig.show()


def _init_splitnet_layer_00(*shape, device="cpu", tanh_gain=1.137, sigmoid_gain=1.65):
    W = torch.randn(*shape, device=device)
    fan_in = W.shape[1]
    fan_out = W.shape[0]
    assert fan_out % 4 == 0

    cs = fan_out // 4  # chunk size
    W[:cs] = torch.randn_like(W[:cs]) * tanh_gain / (fan_in) ** 0.5
    W[cs : cs * 2] = torch.randn_like(W[:cs]) * sigmoid_gain / (fan_in) ** 0.5
    return W


def rootfinding_init(f, cfg, verbose=False):
    from_x, to_x = cfg["search_interval"]
    tol = cfg.get("tol", 1e-3)

    while (to_x - from_x) > tol:
        c = (to_x + from_x) / 2
        f_at_c = f(c)

        if f_at_c > 0:
            to_x = c
        else:
            from_x = c

        # print(f_at_c, to_x - from_x)
        if verbose:
            print(f"f({c:.3f})={f_at_c:.6f} in [{from_x:.3f}, {to_x:.3f}]")

    return c


def get_stds(layer, x, tanh_gain=1.0, sigmoid_gain=1.0):
    layer.linear.weight.data = _init_splitnet_layer_00(*layer.linear.weight.shape, tanh_gain=tanh_gain, sigmoid_gain=sigmoid_gain)

    out, acts = layer.forward_with_activations(x)
    x, preact, preact_tanh, preact_sigmoid, preact_sin, preact_cos, act_tanh, act_sigmoid, act_sin, act_cos = acts

    tanh_act_std, sigmoid_act_std = act_tanh.std(), act_sigmoid.std()
    return tanh_act_std, sigmoid_act_std


class SplitNetManager(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def plot_activations(self, x):
        from IPython.display import display

        out, intermediate_acts = self.net.forward_with_activations(x)
        # x = intermediate_acts[0][0]

        print("Input to the network")
        display(x.plt)
        for layer_i, interim in enumerate(intermediate_acts):
            out, acts = interim
            if len(acts) > 0:
                plot_acts(acts, title=f"layer {layer_i}")
            display(out.plt)

    def init_01(self, x):
        cfg = {
            "search_interval": [0, 20],
        }

        desired_tanh_std, desired_sigmoid_std = 0.5, 0.2

        layers = self.net.net
        if self.net.outermost_linear:
            layers = layers[:-1]

        h = x
        for i, layer in enumerate(layers):
            _, acts = (layer.forward_with_activations(h))
            plot_acts(acts, f'layer{i} before')
            
            layer.linear.bias.data = 1e-6 * torch.randn_like(layer.linear.bias.data)
            f = lambda tanh_gain: get_stds(layer, h, tanh_gain=tanh_gain)[0] - desired_tanh_std
            tanh_gain = rootfinding_init(f, cfg)

            f = lambda sigmoid_gain: get_stds(layer, h, tanh_gain=tanh_gain, sigmoid_gain=sigmoid_gain)[1] - desired_sigmoid_std
            sigmoid_gain = rootfinding_init(f, cfg)

            h, acts = layer.forward_with_activations(h)
            plot_acts(acts, f'layer{i} after')

        return self.net
