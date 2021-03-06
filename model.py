import numpy as np
from typing import List
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from utils import AutoEncoder, Linear
from typing import Any

import numpy as np

class scScope(nn.Module):

    def __init__(
            self,
            input_dim,
            #batch_size,
            #num_inputs,
            encoder_layers_dim: List,
            decoder_layers_dim: List,
            latent_layer_out_dim: int,
            t: int = 2,
            
            **kwargs
        ):

        super(scScope, self).__init__()

        self.t = t
        self.input_dim = input_dim

        self.autoencoder = AutoEncoder(input_dim, encoder_layers_dim, decoder_layers_dim, latent_layer_out_dim,
                                      activation='relu', weight_initializer='normal', weight_init_params={'std': 0.1},
                                      bias_initializer='zeros', batchnorm=False)
        #num_batch = num_inputs//batch_size
        #self.batch_effect_layer = nn.Linear(num_batch, self.input_dim, bias=False)
        #nn.init.zeros_(self.batch_effect_layer.weight)

        impute_layer1 = Linear(self.input_dim, 64, activation='relu',
                                         weight_init='normal', weight_init_params={'std': 0.1},
                                         bias_init='zeros')

        impute_layer2 = Linear(64, self.input_dim, activation=None,
                                         weight_init='normal', weight_init_params={'std': 0.1},
                                         bias_init='zeros')


        self.imputation_model = nn.Sequential(impute_layer1, impute_layer2)
    
    
    def loss_fn(self, y_pred, input_d):
        
        output_layer_list = y_pred
        input_d_corrected = input_d
        val_mask = torch.sign(input_d_corrected)
        for i in range(len(output_layer_list)):
            out_layer = output_layer_list[i]
            if i == 0:
                loss_value = (torch.norm(torch.mul(val_mask, out_layer - input_d_corrected)))
            else:
                loss_value = loss_value + (torch.norm(torch.mul(val_mask, out_layer - input_d_corrected)))
        return loss_value

    def forward(self, X):
       
        latent_features_list = []
        output_list = []
        

        for i in range(self.t):
            if i == 0:
                x = F.relu(X)
            else:
                imputed = self.imputation_model(output)
                imputed = torch.mul(1 - torch.sign(X), imputed)
                x = F.relu(imputed + X)
            latent_features, output = self.autoencoder(x)
            output_list.append(output)
            latent_features_list.append(latent_features)

        return output_list
    
    def step(self, batch: Any, batch_idx: int,) -> Any:
        
        x, y = batch
        batch = (x, batch_idx), y
        output = self.helper_class.step(batch, batch_idx)
        
        return output

    def _func(self, x):
        
        return np.concatenate([i[0][1] for i in x], 0)