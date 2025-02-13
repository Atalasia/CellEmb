import pickle

from torch.utils.data import Dataset

import pandas as pd
import torch

class DrugSmiles:

    def __init__(self, fp):
        self.d_smiles = dict()
        with open(fp, 'rb') as f:
            self.d_smiles = pickle.load(f)


class D2Dataset(Dataset):

    def __init__(self, train_data, file_paths):

        super(D2Dataset, self).__init__()

        self.drug_smiles = DrugSmiles(file_paths["smiles"]).d_smiles
        self.embed_dict = torch.load(file_paths["embeds"])
        self.expr_dict = self.__parse_expressions__(file_paths["exprs"])

        if isinstance(train_data, list):
            self.data = pd.read_csv(train_data[0], sep='\t', header=None, comment="#")
            for idx, fp in enumerate(train_data):
                if idx != 0:
                    temp = pd.read_csv(fp, sep="\t", header=None)
                    self.data = pd.concat([self.data, temp], ignore_index=True)
        else:
            self.data = pd.read_csv(train_data, sep='\t', header=None, comment="#")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        cl_id = self.data.iloc[idx, 0]
        d1_id = self.data.iloc[idx, 1]
        d2_id = self.data.iloc[idx, 2]
        syn_score = self.data.iloc[idx, 3]

        cl_embed = self.embed_dict[cl_id]
        expr = self.expr_dict[cl_id]
        d1_smiles_embed = self.drug_smiles[d1_id]
        d2_smiles_embed = self.drug_smiles[d2_id]

        return cl_id, d1_id, d2_id, cl_embed, expr, syn_score, d1_smiles_embed, d2_smiles_embed

    @staticmethod
    def __parse_expressions__(expr_fp):

        expr = pd.read_csv(expr_fp, sep="\t", header=0, index_col=0)
        expr_dict = {index: torch.tensor(row.tolist(), dtype=torch.float) for index, row in expr.iterrows()}

        return expr_dict
