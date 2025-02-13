from dataset import ContraSet
from contra_model_1024 import DrugResponseEmbedder, GeneExpDimReducer
from contrastive_loss_lw import ContrastiveLoss

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import os, tqdm

from datetime import datetime

import pandas as pd

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

torch.cuda.set_device(0)
torch.cuda.empty_cache()


def train_epoch(model, dl, lf, device, opt):

    dataset_loss_sum = 0.0
    dl_len = len(dl.dataset)

    for batch in dl:

        cl1_id, cl2_id, cl1_exp, cl2_exp, pcc_val, data_count = batch

        cl1_exp = cl1_exp.to(device)
        cl2_exp = cl2_exp.to(device)

        pcc_val = pcc_val.to(device, dtype=torch.float)
        pcc_val = pcc_val.unsqueeze(1)
        
        data_count = data_count.to(device, dtype=torch.float)
        data_count = data_count.unsqueeze(1)

        a_embed, b_embed = model(cl1_exp, cl2_exp)
        contr_loss_batch = lf(a_embed, b_embed, pcc_val, data_count)
        sum_loss = torch.sum(contr_loss_batch)
        avg_loss = torch.mean(contr_loss_batch)
        dataset_loss_sum += sum_loss.item()

        opt.zero_grad()
        avg_loss.backward()
        opt.step()

    return dataset_loss_sum / dl_len


def infer(params, file_paths):

    device = params["device"]
    embedder = GeneExpDimReducer()
    weights = torch.load(params["weights"])
    embeds_fp = params["embeds"]

    expr = pd.read_csv(file_paths["expr"], sep="\t", header=0, index_col=0)
    expr_dict = {index: torch.tensor(row.tolist(), dtype=torch.float) for index, row in expr.iterrows()}

    filter_weights = {i[12:]: weights[i] for i in weights.keys()}

    embedder.load_state_dict(filter_weights)
    embedder.to(device)
    embedder.eval()

    vector_dict = dict()

    for cl_id in list(expr_dict.keys()):
        input = expr_dict[cl_id].unsqueeze(0)
        input = input.to(device)

        out_val = embedder(input).flatten().cpu()
        vector_dict[cl_id] = out_val

    torch.save(vector_dict, embeds_fp)


def train(params, file_paths):

    num_epochs = params['num_epochs']
    device = params["device"]
    learning_rate = params["learning_rate"]

    gdsc_fp = file_paths["gdsc"]
    prism_p_fp = file_paths["prism_p"]
    prism_s_fp = file_paths["prism_s"]
    ctrp_fp = file_paths["ctrp"]

    ds = ContraSet(file_paths["expr"], [gdsc_fp, prism_p_fp, prism_s_fp, ctrp_fp])
    dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0)

    model = DrugResponseEmbedder()
    model.to(device)

    opt = optim.Adam(model.parameters(), lr=learning_rate)
    loss_func = ContrastiveLoss(2.0)

    model.train()

    prev_loss = 9999999.0
    for epoch in tqdm.tqdm(range(num_epochs), desc="train epoch"):
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, learning_rate))

        train_loss = train_epoch(model, dl, loss_func, device, opt)
        print("loss:", train_loss)

        if prev_loss - train_loss < 0.001:
            break
        prev_loss = train_loss

    return model


def main():
    file_paths = {
        "expr": "../data/parsed_24q4_expression.tsv",
        "gdsc": "../data/gdsc_pair_pcc.txt",
        "prism_p": "../data/prism_primary_pair_pcc.txt",
        "prism_s": "../data/prism_secondary_pair_pcc.txt",
        "ctrp": "../data/ctrp_pair_pcc.txt",
    }

    local_time = datetime.now().strftime("%Y%M%d_%H:%M:%S")

    params = {
        "num_epochs": 100,
        "device": torch.device("cuda:0"),
        "weights": "../weights/%s.pt" % local_time,
        "embeds": "../results/%s_embed.pkl" % local_time,
        "learning_rate": 0.0001,
    }

    model = train(params, file_paths)
    torch.save(model.state_dict(), params["weights"])
    infer(params, file_paths)


torch.set_num_interop_threads(15)
torch.set_num_threads(15)
main()
