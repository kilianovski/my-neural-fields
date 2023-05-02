import torch
from torch import nn
import numpy as np
from collections import OrderedDict
import math
import torch.nn.functional as F

import re
from collections import OrderedDict

def get_subdict(dictionary, key=None):
    if dictionary is None:
        return None
    if (key is None) or (key == ''):
        return dictionary
    key_re = re.compile(r'^{0}\.(.+)'.format(re.escape(key)))
    return OrderedDict((key_re.sub(r'\1', k), value) for (k, value)
        in dictionary.items() if key_re.match(k) is not None)


class BatchLinear(nn.Linear, nn.Module):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output


class Sine(nn.Module):
    def __init(self):
        super().__init__()

    def forward(self, input):
        # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
        return torch.sin(30 * input)


class FCBlock(nn.Module):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''

    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None):
        super().__init__()

        self.first_layer_init = None

        nl, nl_weight_init, first_layer_init = (Sine(), sine_init, first_layer_sine_init)

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init

        self.net = []
        self.net.append(nn.Sequential(
            BatchLinear(in_features, hidden_features), nl
        ))

        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                BatchLinear(hidden_features, hidden_features), nl
            ))

        if outermost_linear:
            self.net.append(nn.Sequential(BatchLinear(hidden_features, out_features)))
        else:
            self.net.append(nn.Sequential(
                BatchLinear(hidden_features, out_features), nl
            ))

        self.net = nn.Sequential(*self.net)
        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None: # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, coords):
        output = self.net(coords)
        return output
    
import torch.nn.init as init
    
class SplitLayer(nn.Module):
    def __init__(self, input_dim, h=32, activation_type="default", is_last=False):
        super().__init__()
        self.linear = nn.Linear(input_dim, h*4)
        self.activation_type = activation_type
        # self.init_weights()
        self.is_last = is_last
        
    # def init_weights(self):
    #     if self.activation_type == "default":
    #         init.xavier_uniform_(self.linear.weight)
    #     elif self.activation_type == "siren":
    #         init.normal_(self.linear.weight, mean=0.0, std=1.0 / self.linear.in_features)
            
    def init_weights(self):
        # Apply Xavier initialization for sigmoid and tanh
        init.xavier_uniform_(self.linear.weight[:, :2 * self.linear.out_features // 4])
        # Apply SIREN initialization for sin and cos
        init.normal_(self.linear.weight[:, 2 * self.linear.out_features // 4:], mean=0.0, std=1.0 / self.linear.in_features)# np.sqrt(6 / num_input) / 30
        # init.normal_(self.linear.weight[:, 2 * self.linear.out_features // 4:], mean=0.0, std=np.sqrt(6 / self.linear.in_features) / 10)
        
    def forward(self, x):
        o = self.linear(x).chunk(4,-1)
        if self.activation_type == "default":
            o = o[0].sigmoid() * o[1].tanh() * o[2].sin() * o[3].cos()
        elif self.activation_type == "siren":
            o = o[0].sigmoid() * o[1].tanh() * o[2].sin() * o[3].cos()
            # o = o[0].sin() * o[1].cos() * o[2].sigmoid() * o[3].tanh()
        return self.post_process(o, x)
    
    def post_process(self, x, inp):
        if not self.is_last:
            x = x + inp
            # x = x
        # else:
        #     x = x + inp
        return x
    
    def forward_with_activations(self, x):
        preact = self.linear(x)
        
        if self.activation_type == "default":
            preact_sigmoid, preact_tanh, preact_sin, preact_cos = preact.chunk(4, dim=-1)
            act_sigmoid, act_tanh, act_sin, act_cos = preact_sigmoid.sigmoid(), preact_tanh.tanh(), preact_sin.sin(), preact_cos.cos()
        elif self.activation_type == "siren":
            preact_sin, preact_cos, preact_sigmoid, preact_tanh = preact.chunk(4, dim=-1)
            act_sin, act_cos, act_sigmoid, act_tanh = preact_sin.sin(), preact_cos.cos(), preact_sigmoid.sigmoid(), preact_tanh.tanh()
        
        h = act_tanh * act_sigmoid * act_sin * act_cos
        
        h = self.post_process(h, x)
        
        return h, [x, preact, preact_tanh, preact_sigmoid, preact_sin, preact_cos, act_tanh, act_sigmoid, act_sin, act_cos]

    
class UberSDF(nn.Module):
    def __init__(self, in_features, out_features, num_hidden_layers, hidden_features,
                 outermost_linear=False, nonlinearity='relu', weight_init=None, activation_type="siren"):
        super(UberSDF, self).__init__()
        
        self.net = []
        self.net.append(nn.Sequential(
                SplitLayer(hidden_features, hidden_features, activation_type)#, BatchNorm1dNCSpatial(hidden_features)
            ))
        
        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                SplitLayer(hidden_features, hidden_features, activation_type)#, BatchNorm1dNCSpatial(hidden_features)
            ))
            
        self.net.append(SplitLayer(hidden_features, out_features, activation_type, is_last=True))
        
        self.net = nn.Sequential(*self.net)
        
        import rff
        encoded_size = hidden_features//2
        self.encoding = rff.layers.GaussianEncoding(sigma=10.0, input_size=3, encoded_size=encoded_size)
        
        
    def forward(self, x):
        x = self.encoding(x)
        
        for layer in self.net:
            x = layer(x)
        
        return x
    
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

def normal_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        # Initialize the weights with a normal distribution
        nn.init.normal_(m.weight, mean=0.0, std=0.5)
        
        # Initialize the biases with zeros
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    

class SingleBVPNet(nn.Module):
    '''A canonical representation network for a BVP.'''

    def __init__(self, out_features=1, type='sine', in_features=2,
                 mode='mlp', hidden_features=256, num_hidden_layers=3, **kwargs):
        super().__init__()
        self.mode = mode
        
        if type=='sine':
            self.net = FCBlock(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                               hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
        elif type=='split_act':
            self.net = UberSDF(in_features=in_features, out_features=out_features, num_hidden_layers=num_hidden_layers,
                               hidden_features=hidden_features, outermost_linear=True, nonlinearity=type)
            # self.net.apply(normal_init)
        else:
            raise "Choose the model type!"
            

    def forward(self, model_input, params=None):
        # Enables us to compute gradients w.r.t. coordinates
        coords = model_input['coords'].clone().detach().requires_grad_(True)

        output = self.net(coords)
        return {'model_in': coords, 'model_out': output}
    
    def forward_with_activations(self, model_input):
        x = model_input['coords'].clone().detach().requires_grad_(True)
        
        if isinstance(self.net, UberSDF):
            x, acts = self.net.forward_with_activations(x)
        else:
            x = self.net(x)
            acts = []

        return x, acts


def sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)
            
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
        display(x['coords'].plt)
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