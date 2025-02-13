import pandas as pd
import torch
from torch.utils.data import Dataset

class ContraSet(Dataset):

    def __init__(self, exprs_fp, data_fps):

        super(ContraSet, self).__init__()

        self.expr_dict = self.__parse_expressions__(exprs_fp)

        if isinstance(data_fps, list):
            self.data = pd.read_csv(data_fps[0], sep='\t', header=None, comment="#")
            for idx, fp in enumerate(data_fps):
                if idx != 0:
                    temp = pd.read_csv(fp, sep="\t", header=None)
                    self.data = pd.concat([self.data, temp], ignore_index=True)
        else:
            self.data = pd.read_csv(data_fps, sep='\t', header=None, comment="#")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        cl1_id = self.data.iloc[idx, 0]
        cl2_id = self.data.iloc[idx, 1]
        pcc_val = float(self.data.iloc[idx, 2])
        data_count = int(self.data.iloc[idx, 3])

        return cl1_id, cl2_id, self.expr_dict[cl1_id], self.expr_dict[cl2_id], pcc_val, data_count

    @staticmethod
    def __parse_expressions__(expr_fp):

        expr = pd.read_csv(expr_fp, sep="\t", header=0, index_col=0)
        expr_dict = {index: torch.tensor(row.tolist(), dtype=torch.float) for index, row in expr.iterrows()}

        return expr_dict
