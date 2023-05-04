import torch
from torch import nn
import torch.nn.functional as F

import os
import math


class SplitLayer(nn.Module):
    def __init__(self, input_dim, output_dim, m=1., omega=30.):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 4)
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
        if not hasattr(m, '__len__'):
            m = [m] * (hidden_layers+2)
        self.encoding = rff.layers.GaussianEncoding(sigma=10.0, input_size=in_features, encoded_size=encoding_size)
        self.net = []
        layer_cls = SplitLayerLin
        self.net += [layer_cls(encoding_size*2, hidden_features, m=m[0])]
        self.net += [layer_cls(hidden_features, hidden_features, m=m[i+1]) for i in range(hidden_layers)]

        if outermost_linear:
            self.net += [nn.Linear(hidden_features, out_features)]
        else:
            self.net += [layer_cls(hidden_features, out_features, m=m[-1])]

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
                 m=1.,
                 init_fn=None,
                 ):
        super().__init__()
        if not hasattr(m, '__len__'):
            m = [m] * (hidden_layers+2)

        import rff
        self.encoding = rff.layers.GaussianEncoding(sigma=10.0, input_size=in_features, encoded_size=encoding_size)
        self.net = []

        self.net += [SplitLayer(encoding_size*2, hidden_features, m=m[0])]
        self.net += [SplitLayer(hidden_features, hidden_features, m=m[i+1]) for i in range(hidden_layers)]

        if outermost_linear:
            self.net += [nn.Linear(hidden_features, out_features)]
        else:
            self.net += [SplitLayer(hidden_features, out_features, m=m[-1])]

        self.net = nn.Sequential(*self.net)
        self.outermost_linear = outermost_linear
        
        if init_fn == "my_init_02":
            my_init_02(self)
        elif init_fn == 'my_init_03':
            my_init_03(self)
        elif init_fn is not None:
            raise ArgumentError(f"Unknown init_fn: {init_fn}")

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
    if len(a) > 1000:
        a = np.random.choice(a, size=1_000, replace=False)
    return a


def U(*shape):
    x = torch.rand(*shape)
    x = (x - 0.5) * 2.0
    return x

def plot_distributions(data, title='sample titles'):
    for k, v in data.items():
        print(k, tensor2str(v))
    data = {k: to_np(v) for k, v in data.items()}

    df = pd.DataFrame(data)

    # Melt the dataframe to a long format for easier plotting
    df_melted = df.melt(var_name="act", value_name="value")

    # Plot histograms using Plotly Express
    fig = px.histogram(
        df_melted,
        x="value",
        color="act",
        facet_col="act",
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

def tensor2str(x):
    return f'{tuple(x.shape)} in [{x.min():.3f}, {x.max():.3f}] μ={x.mean():.3f} σ={x.std():.3f}'

def plot_acts(acts, title="acts"):
    x, preact, preact_tanh, preact_sigmoid, preact_sin, preact_cos, act_tanh, act_sigmoid, act_sin, act_cos = acts

    # Create a dataframe from your arrays
    data = {
        # "preact_tanh": preact_tanh,
        # "preact_sigmoid": preact_sigmoid,
        # "preact_sin": preact_sin,
        # "preact_cos": preact_cos,
        f"act_tanh": act_tanh,
        f"act_sigmoid": act_sigmoid,
        f"act_sin": act_sin,
        f"act_cos": act_cos,
    }
    
    

    plot_distributions(data, title=title)



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


def my_init_02(model):
    def U(*shape, r=1):
        x = torch.rand(*shape)
        
        x = (x-0.5)*(2*r)
        return x
    

    x = U(2000, 2)
    
    for i, (layer) in enumerate(model.net):
        torch.nn.init.zeros_(layer.linear.bias)

        W = layer.linear.weight.data

        s, n = W.shape
        s = s // 4
        c = 0.1
        torch.nn.init.uniform_(W[:s], -c, c)
        
        c = c * 1.5
        torch.nn.init.uniform_(W[s:s*2], -c, c)
        
        if i == len(model.net) - 1:
            c = (0.1)/10
            torch.nn.init.uniform_(W[:s], -c, c)
        
            c = (c * 1.5)/10
            torch.nn.init.uniform_(W[s:s*2], -c, c)

        # print(layer.linear.weight.data)
        # break
        print(layer.linear.weight.data.std())
        
    
    o, intermediate_acts = model.forward_with_activations(x)
    for i, (h, acts) in enumerate(intermediate_acts):
        plot_acts(acts, title=f'Layer {i}')

def init_siren(W, fan_in, omega=30, init_c=24, flic=2, is_first=False):
    if is_first:
        c = flic / fan_in
    else:
        c = np.sqrt(init_c / fan_in) / omega    
    W.uniform_(-c, c)


def my_init_03(model):
    for i, (layer) in enumerate(model.net):
        is_first = i == 0
        
        torch.nn.init.zeros_(layer.linear.bias)

        W = layer.linear.weight.data

        s, fan_in = W.shape
        s = s // 4
        c = 0.1

        torch.nn.init.uniform_(W[:s], -c, c)

        c = c * 1.5
        torch.nn.init.uniform_(W[s:s*2], -c, c)

        init_siren(W[s*2:], fan_in=fan_in, is_first=False)
        print(layer.linear.weight.data)

    # last
    layer.linear.weight.data /= 30

    x = U(2000, 2)

    o, intermediate_acts = model.forward_with_activations(x)
    for i, (h, acts) in enumerate(intermediate_acts):
        plot_acts(acts, title=f'Layer {i}')


import torch
from torch import nn
import torch.nn.functional as F

import os
import math


class SplitLayerOmegas(nn.Module):
    def __init__(self, input_dim, output_dim, m=1., omegas=(1,1,1.,1)):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 4)
        self.m = m
        self.omegas = omegas

    def forward(self, x):
        h, acts = self.forward_with_activations(x)
        return h

    def forward_with_activations(self, x):
        preact = self.linear(x)
        preacts = preact.chunk(4, dim=-1)
        preacts = list(preacts)
    
        for i in range(len(preacts)):
            preacts[i] = self.omegas[i] * preacts[i]

        preact_tanh, preact_sigmoid, preact_sin, preact_cos = preacts
        act_tanh, act_sigmoid, act_sin, act_cos = preact_tanh.tanh(), preact_sigmoid.sigmoid(),preact_sin.sin(), preact_cos.cos()
        h = act_tanh * act_sigmoid * act_sin * act_cos
        
        h = h * self.m

        return h, [x, preact, preact_tanh, preact_sigmoid, preact_sin, preact_cos, act_tanh, act_sigmoid, act_sin, act_cos]

class SplitNetPosEncOmegas(nn.Module):
    def __init__(self, 
                 in_features=2,
                 encoding_size=64,
                 hidden_features=64, 
                 hidden_layers=1, 
                 out_features=1, 
                 outermost_linear=True,
                 m=1.,
                 omegas=(1,1,1.,1),
                 init_fn=None,
                 ):
        super().__init__()
        if not hasattr(m, '__len__'):
            m = [m] * (hidden_layers+2)
            
        is_layerwise_omegas = hasattr(omegas[0], '__len__')
        
        if not is_layerwise_omegas:
            omegas = [omegas] * (hidden_layers+2)
            

        import rff
        self.encoding = rff.layers.GaussianEncoding(sigma=10.0, input_size=in_features, encoded_size=encoding_size)
        self.net = []

        self.net += [SplitLayerOmegas(encoding_size*2, hidden_features, m=m[0], omegas=omegas[0])]
        self.net += [SplitLayerOmegas(hidden_features, hidden_features, m=m[i+1], omegas=omegas[i+1]) for i in range(hidden_layers)]

        if outermost_linear:
            self.net += [nn.Linear(hidden_features, out_features)]
        else:
            self.net += [SplitLayerOmegas(hidden_features, out_features, omegas=omegas[-1], m=m[-1])]

        self.net = nn.Sequential(*self.net)
        self.outermost_linear = outermost_linear
        self.omegas = omegas
        
        if init_fn == "my_init_02":
            my_init_02(self)
        elif init_fn == 'my_init_03':
            my_init_03(self)
            
        elif init_fn == 'giga_init':
            giga_init(self, self.omegas)

        elif init_fn is not None:
            raise ArgumentError(f"Unknown init_fn: {init_fn}")

    def forward(self, x):
        x = self.encoding(x)
        return self.net(x)

    def forward_with_activations(self, x):
        x = self.encoding(x)
        h = x
        intermediate_acts = []

        for layer in self.net:
            if isinstance(layer, SplitLayerOmegas):
                h, acts = layer.forward_with_activations(h)
            else:
                h = layer(h)
                acts = []

            intermediate_acts.append((h, acts))

        return h, intermediate_acts


def giga_init(model, omegas):
    print("WELCOME TO THE GIGA INIT")
    for i, (layer) in enumerate(model.net):
        is_first = i == 0

        torch.nn.init.zeros_(layer.linear.bias)

        W = layer.linear.weight.data

        s, fan_in = W.shape
        s = s // 4
        c = 0.1 / omegas[i][0]

        torch.nn.init.uniform_(W[:s], -c, c)

        c = 0.1 * 1.5 / omegas[i][1]
        torch.nn.init.uniform_(W[s:s*2], -c, c)

        init_siren(W[s*2:], fan_in=fan_in, is_first=False, omega=omegas[i][2])
        print(layer.linear.weight.data)

    x = U(2000, 2)
    o, intermediate_acts = model.forward_with_activations(x)

    o.mean().backward()

    for i, layer in enumerate(model.net):
        W = layer.linear.weight
        G = W.grad.detach().cpu()

        G_tanh, G_sigmoid, G_sin, G_cos = G.chunk(4, dim=0)

        plot_data = {
            'G_tanh': G_tanh, 
            'G_sigmoid': G_sigmoid, 
            'G_sin': G_sin, 
            'G_cos': G_cos
        }
        
        # for k,v in plot_data.items():
        #     print(k, v)
        
        plot_distributions(plot_data, title=f'Layer {i}')

        # if i >= 1:
        #     break