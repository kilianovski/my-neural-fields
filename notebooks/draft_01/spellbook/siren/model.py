import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30, init_c=6, first_layer_init_c=1):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights(init_c=init_c, first_layer_init_c=first_layer_init_c)

    def init_weights(self, init_c, first_layer_init_c):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-first_layer_init_c / self.in_features, first_layer_init_c / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(init_c / self.in_features) / self.omega_0, np.sqrt(init_c / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, first_omega_0=30, hidden_omega_0=30.0, init_c=6, first_layer_init_c=1):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0, first_layer_init_c=first_layer_init_c))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0, init_c=init_c))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(init_c / hidden_features) / hidden_omega_0, np.sqrt(init_c / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output

    def forward_with_activations(self, coords, retain_grad=False):
        """Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!"""
        activations = OrderedDict()

        activation_count = 0
        x = coords
        activations["input"] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations[f"layer_{i}_preact"] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations[f"layer_{i}_act"] = x
            activation_count += 1

        return activations


class CosineLayer(SineLayer):
    def forward(self, input):
        return torch.cos(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.cos(intermediate), intermediate


class Coren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, first_omega_0=30, hidden_omega_0=30.0, init_c=6, first_layer_init_c=1):
        super().__init__()

        self.net = []
        self.net.append(CosineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0, first_layer_init_c=first_layer_init_c))

        for i in range(hidden_layers):
            self.net.append(CosineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0, init_c=init_c))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(init_c / hidden_features) / hidden_omega_0, np.sqrt(init_c / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(CosineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output

    def forward_with_activations(self, coords, retain_grad=False):
        """Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!"""
        activations = OrderedDict()

        activation_count = 0
        x = coords
        activations["input"] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations["_".join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations["_".join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


def init_siren(W, fan_in, omega=30, init_c=24, flic=2, is_first=False):
    if is_first:
        c = flic / fan_in
    else:
        c = np.sqrt(init_c / fan_in) / omega
    W.uniform_(-c, c)


class SplitLayer(nn.Module):
    def __init__(self, input_dim, output_dim, m=1.0, omegas=(1, 1, 30, 30), use_bias=True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 4, bias=use_bias)
        self.dropout = nn.Dropout(0)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.m = m
        self.omegas = omegas

        self.init_weights()

    def init_weights(self):
        s = self.output_dim
        fan_in = self.input_dim

        W = self.linear.weight.data
        self.linear.bias.data.uniform_(0, 0)
    #     c = np.sqrt(1 / fan_in) / self.omegas[0]
    #     # print('the c', c)
    #     W[:s].uniform_(-c, c)
    #     # init_siren(W[:s], init_c=1, fan_in=fan_in, is_first=False, omega=self.omegas[0])

    #     init_siren(W[s:s*2],init_c=6, fan_in=fan_in, is_first=False, omega=self.omegas[1])
        init_siren(W[s*2:], fan_in=fan_in, is_first=False, omega=30)

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
        act_tanh, act_sigmoid, act_sin, act_cos = preact_tanh.tanh(), preact_sigmoid.sigmoid(), preact_sin.sin(), preact_cos.cos()
        h = act_tanh * act_sigmoid * act_sin * act_cos

        h = h * self.m

        return h, [x, preact, preact_tanh, preact_sigmoid, preact_sin, preact_cos, act_tanh, act_sigmoid, act_sin, act_cos]


class SirenWithSplit(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, first_omega_0=30, hidden_omega_0=30.0, init_c=6, first_layer_init_c=1):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, is_first=True, omega_0=first_omega_0, first_layer_init_c=first_layer_init_c))

        for i in range(hidden_layers - 1):
            self.net.append(
                SineLayer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0, init_c=init_c)
                )
            
        self.net.append(
                SplitLayer(hidden_features, hidden_features//4)
                )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features//4, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(init_c / hidden_features) / hidden_omega_0, np.sqrt(init_c / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        output = self.net(coords)
        return output

    def forward_with_activations(self, coords, retain_grad=False):
        """Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!"""
        activations = OrderedDict()

        activation_count = 0
        x = coords
        activations["input"] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations[f"layer_{i}_preact"] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations[f"layer_{i}_act"] = x
            activation_count += 1

        return activations