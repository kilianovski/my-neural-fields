import torch
from torch import nn
import torch.nn.functional as F

import os
import math
import numpy as np

from train_pipeline import *


def init_siren(W, fan_in, omega=30, init_c=24, flic=2, is_first=False):
    if is_first:
        c = flic / fan_in
    else:
        c = np.sqrt(init_c / fan_in) / omega
    W.uniform_(-c, c)


def _init(W, c):
    W.uniform_(-c, c)


class SplitLayer(nn.Module):
    def __init__(self, input_dim, output_dim, m=1.0, cs=(1, 1, 1, 1), omegas=(1, 1, 1.0, 1), use_bias=True):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 4, bias=use_bias)
        self.dropout = nn.Dropout(0)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.m = m
        self.omegas = omegas
        self.cs = cs

        self.init_weights()

    def init_weights(self):
        self.linear.bias.data.uniform_(0, 0)
        s = self.output_dim
        W = self.linear.weight.data
        _init(W[:s], self.cs[0])
        _init(W[s : s * 2], self.cs[1])
        _init(W[s * 2 : s * 3], self.cs[2])
        _init(W[s * 3 : s * 4], self.cs[3])

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


class SimpleSplitNet(nn.Module):
    def __init__(self, cs, use_bias=True, omegas=(1, 1, 1.0, 1), m=1.0):
        super().__init__()

        in_features = 128
        hidden_layers = 2

        if not hasattr(m, "__len__"):
            m = [m] * (len(hidden_layers) + 1)

        is_layerwise_omegas = hasattr(omegas[0], "__len__")

        if not is_layerwise_omegas:
            omegas = [omegas] * (len(hidden_layers) + 1)

        net = [SplitLayer(in_features, 64, use_bias=use_bias, cs=cs[0], m=m[0], omegas=omegas[0]), SplitLayer(64, 32, use_bias=use_bias, cs=cs[1], m=m[1], omegas=omegas[1]), nn.Linear(32, 3)]

        _init(net[-1].weight.data, cs[2])
        self.net = nn.Sequential(*net)

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


def get_example_model():
    kwargs = {"cs": [(1, 1, 1, 1), (1, 1, 1, 1), 0.1], "omegas": [(1, 1, 1, 1), (1, 1, 1, 1)], "m": [1, 1]}

    net = SimpleSplitNet(**kwargs)


import wandb

m_range = (0.1, 30)
c_range = (1e-3, 1e1)
omega_range = (0.1, 30)
PROJECT_NAME = "splitnet_3_sweep"

sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "psnr"},
    "parameters": {
        **{f"m{i}": {"distribution": "uniform", "min": m_range[0], "max": m_range[1]} for i in range(2)},
        **{f"c{i}": {"distribution": "uniform", "min": c_range[0], "max": c_range[1]} for i in range(4 + 4 + 1)},
        **{f"omega{i}": {"distribution": "uniform", "min": omega_range[0], "max": omega_range[1]} for i in range(4 + 4)},
        "lr": {"values": [1e-3, 1e-4, 1e-5]},
        "weight_decay": {'values': [0, 1e-5]},
    },
}


def _train_for_sweep(model, cfg, lr, weight_decay):
    seed_all(cfg["random_seed"])
    device = cfg["device"]
    total_steps = cfg["total_steps"]

    model_input, ground_truth, H, W = load_data(cfg)
    model_input, ground_truth = model_input.to(device), ground_truth.to(device)

    model.to(device)

    import rff
    encoding = rff.layers.GaussianEncoding(sigma=10.0, input_size=2, encoded_size=64).to(device)
    model_input = encoding(model_input)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for step in range(total_steps):
        model_output = model(model_input)
        mse, psnr = mse_and_psnr(model_output, ground_truth)
        loss = mse

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return psnr.item()


def objective(c):
    cs = [(c["c0"], c["c1"], c["c2"], c["c3"]), (c["c4"], c["c5"], c["c6"], c["c7"]), c["c8"]]
    omegas = [(c["omega0"], c["omega1"], c["omega2"], c["omega3"]), (c["omega4"], c["omega5"], c["omega6"], c["omega7"])]
    m = [c["m0"], c["m1"]]
    kwargs = {"cs": cs, "omegas": omegas, "m": m}
    net = SimpleSplitNet(**kwargs)
    psnr = _train_for_sweep(net, cfg, c["lr"], c["weight_decay"])
    return psnr


def main():
    wandb.init(project=PROJECT_NAME)
    psnr = objective(wandb.config)
    wandb.log({"psnr": psnr})


import os
from hydra import initialize, initialize_config_module, initialize_config_dir, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf


def load_cfg(config_name="config", overrides=()):
    # with initialize_config_dir(config_dir="/app/notebooks/draft_02/conf"):
    with initialize(version_base=None, config_path="./conf"):
        cfg = compose(config_name=config_name, overrides=list(overrides))
        return cfg

cfg = load_cfg("sweep_config_0", overrides=["+device=cuda:0"])
print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=PROJECT_NAME)

    wandb.agent(sweep_id, function=main, count=10)
