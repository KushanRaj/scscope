from torch.utils.data import Dataset
import numpy as np
import torch

class GeneExpression(Dataset):

    def __init__(self, filepath):

        ext = filepath.split('.')[-1]

        assert ext in ['csv', 'npz', 'npy'], f'Given file is of wrong type. Please ensure data is formatted correctly as one of the following | {" | ".join(["csv", "npz", "npy"])} |'

        if ext == 'csv':

            data = np.loadtxt(filepath, delimiter=',')

        else:

            data = np.load(filepath)

        print(f'{filepath} loaded, Data Shape : {data.shape}\nAssuming {data.shape[0]} Cells/Patients and {data.shape[1]} Genes')

        self.data = data

    def __len__(self):

        return self.data.shape[0]

    def __getitem__(self, id):

        return torch.from_numpy(self.data[id]).float()



    