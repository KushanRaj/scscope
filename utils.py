from typing import Dict, List
import torch.nn as nn  
import torch
import yaml
import os
import sys
import numpy as np

class ExponentialActivation(nn.Module):

    def __init__(self):
        super(ExponentialActivation, self).__init__()

    def forward(self, x):
        x = torch.exp(x)
        return x


class Linear(nn.Module):

    activations = {'relu': nn.ReLU(), 'sigmoid': nn.Sigmoid(), 'softmax': nn.Softmax(),
                   'exp': ExponentialActivation(), 'softplus': nn.Softplus()}
    initializers = {'xavier': nn.init.xavier_uniform_, 'zeros': nn.init.zeros_, 'normal': nn.init.normal_}

    def __init__(
            self,
            input_dim,
            out_dim,
            batchnorm = False,
            activation = 'relu',
            dropout = 0.,
            weight_init = None,
            weight_init_params: Dict = None,
            bias_init = None,
            bias_init_params: Dict = None
    ):
        super(Linear, self).__init__()

        if weight_init_params is None:
            weight_init_params = {}
        if bias_init_params is None:
            bias_init_params = {}
        self.batchnorm_layer = None
        self.act_layer = None
        self.dropout_layer = None
        
        self.linear = nn.Linear(input_dim, out_dim)
        
        if weight_init is not None:
            self.initializers[weight_init](self.linear.weight, **weight_init_params)
        if bias_init is not None:
            self.initializers[bias_init](self.linear.bias, **bias_init_params)

        if batchnorm:
            self.batchnorm_layer = nn.BatchNorm1d(out_dim)
        if activation:
            self.act_layer = self.activations[activation]
        if dropout > 0.0:
            self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear(x)
        if self.batchnorm_layer:
            x = self.batchnorm_layer(x)
        if self.act_layer:
            x = self.act_layer(x)
        if self.dropout_layer:
            x = self.dropout_layer(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(
            self,
            input_d,
            encoder_layers_dim: List,
            decoder_layers_dim: List,
            latent_layer_out_dim: int,
            batchnorm: bool = True,
            activation: str = 'relu',
            dropout: float = 0.,
            weight_initializer=None,
            weight_init_params: Dict = None,
            bias_initializer = None,
            bias_init_params: Dict = None,
            return_output = True,
            **kwargs
    ):
        super(AutoEncoder, self).__init__()
        if weight_init_params is None:
            weight_init_params = {}
        if bias_init_params is None:
            bias_init_params = {}

        self.input_dim = input_d
        self.batchnorm = batchnorm
        encode_layers = []
        if len(encoder_layers_dim) > 0:
            for i in range(len(encoder_layers_dim)):
                if i == 0:
                    encode_layers.append(Linear(self.input_dim, encoder_layers_dim[i],
                                                          batchnorm=self.batchnorm, activation=activation,
                                                          dropout=dropout,
                                                          weight_init=weight_initializer,
                                                          weight_init_params=weight_init_params,
                                                          bias_init=bias_initializer, bias_init_params=bias_init_params))
                else:
                    encode_layers.append(Linear(encoder_layers_dim[i - 1], encoder_layers_dim[i],
                                                          batchnorm=self.batchnorm, activation=activation,
                                                          dropout=dropout,
                                                          weight_init=weight_initializer,
                                                          weight_init_params=weight_init_params,
                                                          bias_init=bias_initializer, bias_init_params=bias_init_params))
            self.latent_layer_input_dim = encoder_layers_dim[-1]
        else:
            self.latent_layer_input_dim = self.input_dim
        self.encode = nn.Sequential(*encode_layers)

        self.latent_layer = Linear(self.latent_layer_input_dim, latent_layer_out_dim,
                                             batchnorm=self.batchnorm, activation=activation,
                                             dropout=dropout,
                                             weight_init=weight_initializer, weight_init_params=weight_init_params,
                                             bias_init=bias_initializer, bias_init_params=bias_init_params)

        decode_layers = []
        if len(decoder_layers_dim) > 0:
            for i in range(len(decoder_layers_dim)):
                if i == 0:
                    decode_layers.append(Linear(latent_layer_out_dim, decoder_layers_dim[i],
                                                          batchnorm=self.batchnorm, activation=activation,
                                                          dropout=dropout,
                                                          weight_init=weight_initializer,
                                                          weight_init_params=weight_init_params,
                                                          bias_init=bias_initializer, bias_init_params=bias_init_params))
                else:
                    decode_layers.append(Linear(decoder_layers_dim[i - 1], decoder_layers_dim[i],
                                                          batchnorm=self.batchnorm, activation=activation,
                                                          dropout=dropout,
                                                          weight_init=weight_initializer,
                                                          weight_init_params=weight_init_params,
                                                          bias_init=bias_initializer, bias_init_params=bias_init_params))
            self.output_layer_input_dim = decoder_layers_dim[-1]
        else:
            self.output_layer_input_dim = latent_layer_out_dim
        self.decode = nn.Sequential(*decode_layers)
        self.return_output = return_output
        if return_output:
            self.output_layer = Linear(self.output_layer_input_dim, self.input_dim,
                                             activation=activation,
                                             weight_init=weight_initializer, weight_init_params=weight_init_params,
                                             bias_init=bias_initializer, bias_init_params=bias_init_params)
                

    def forward(self, x):
        encoded = self.encode(x)
        latent = self.latent_layer(encoded)
        output = self.decode(latent)
        if self.return_output:
            output = self.output_layer(output)

        return latent, output


def read_yaml(config_path):
    if not os.path.exists(config_path):
        raise ValueError(f"{config_path} does not exist.")
    config = yaml.safe_load(open(config_path, "r"))
    
    return config

class AvgMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {}

    def add(self, batch_metrics):
        if self.metrics == {}:
            for key, value in batch_metrics.items():
                self.metrics[key] = [value]
        else:
            for key, value in batch_metrics.items():
                self.metrics[key].append(value)

    def get(self):
        return {key: np.mean(value) for key, value in self.metrics.items()}

    def msg(self):
        avg_metrics = {key: np.mean(value) for key, value in self.metrics.items()}
        return "".join(["[{}] {:.5f} ".format(key, value) for key, value in avg_metrics.items()])


def pbar(p=0, msg="", bar_len=20):
    sys.stdout.write("\033[K")
    sys.stdout.write("\x1b[2K" + "\r")
    block = int(round(bar_len * p))
    text = "Progress: [{}] {}% {}".format(
        "\x1b[32m" + "=" * (block - 1) + ">" + "\033[0m" + "-" * (bar_len - block),
        round(p * 100, 2),
        msg,
    )
    print(text, end="\r")
    if p == 1:
        print()