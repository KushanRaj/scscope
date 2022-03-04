from pathlib import Path
import torch
import sys
import torch.nn as nn
from model import scScope
from utils import read_yaml, pbar, AvgMeter
from dataset import GeneExpression
from torch.utils.data import DataLoader
import os
import numpy as np
from datetime import datetime

OPTIM = {'adam' : torch.optim.Adam, 'sgd' : torch.optim.SGD }

class Trainer:

    def __init__(
                 self, 
                 config_path : Path,
                ):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        config = read_yaml(config_path)
            
        model_args = config['model_args'] 

        self.model = scScope(**model_args).to(self.device)

        self.output = f"{config['output']}/{datetime.now().strftime('%Y-%m-%d_%H-%M')}"

        os.makedirs(self.output, exist_ok = True)

        data = GeneExpression(**config['data_args'] )

        self.metric_meter = AvgMeter()

        self.loader = DataLoader(data, **config['data_loader']) 

        self.config = config

        if config.get('optim_name', False):
            self.optim = OPTIM[config['optim_name'].lower()](self.model.parameters(), **config['optim_args'])

    
    def train_epoch(self):

        self.metric_meter.reset()
        self.model.train()

        for indx, data in enumerate(self.loader):

            data = data.to(self.device)

            output = self.model(data)

            loss = self.model.loss_fn(output, data)

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            self.metric_meter.add({'train_loss' : loss.item()})

            pbar(indx / len(self.loader), msg=self.metric_meter.msg())

        pbar(1, msg=self.metric_meter.msg())

    @torch.no_grad()
    def pred(self):

        self.model.eval()
        outs = []
        for indx, data in enumerate(self.loader):

            data = data.to(self.device)

            output = self.model(data)
            outs.append(output[-1])

            pbar(indx / len(self.loader), msg=self.metric_meter.msg())

        pbar(1, msg=self.metric_meter.msg())


        return torch.cat(outs, 0).cpu().detach().numpy()

    def train(self):
        best_train = float("inf")

        for epoch in range(self.config['epochs']):
            
            print(f"\nepoch: {epoch}")
            print("---------------")

            self.train_epoch()

            
            train_metrics = self.metric_meter.get()
            if train_metrics["train_loss"] < best_train:
                print(
                    "\x1b[34m"
                    + f"train loss improved from {round(best_train, 5)} to {round(train_metrics['train_loss'], 5)}"
                    + "\033[0m"
                )
                best_train = train_metrics["train_loss"]
                torch.save(
                        self.model.state_dict(),
                        os.path.join(self.output, f"best.ckpt"),
                    )
            msg = f"epoch: {epoch}, last train: {round(train_metrics['train_loss'], 5)}, best train: {round(best_train, 5)}"

            
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optim": self.optim.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(self.output, "last.ckpt"),
            )

if __name__ == "__main__":


    trainer = Trainer('config.yaml')

    trainer.train()

    pred = trainer.pred()

    np.savetxt('out.csv', pred, delimiter=',')