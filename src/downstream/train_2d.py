from dataset_2d import D2Dataset
from model_2d import Regressor

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

import os
import tqdm
from datetime import datetime

def pearson(x, y):
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)

    num = torch.sum((x - x_mean) * (y - y_mean))
    denom = torch.sqrt(torch.sum((x - x_mean) ** 2)) * torch.sqrt(torch.sum((y - y_mean) ** 2))

    pcc = num / denom

    return pcc.item()


def train_epoch(model, loss_func, dl, device, opt):

    loss_sum = 0.0
    dl_len = len(dl.dataset)

    for batch in dl:

        cl_id, d1_id, d2_id, cl_embed, exprs, syn_score, d1_smiles, d2_smiles = batch

        d1_smiles = d1_smiles.to(device)
        d2_smiles = d2_smiles.to(device)
        cl_embed = cl_embed.to(device)
        exprs = exprs.to(device)
        syn_score = syn_score.to(device, dtype=torch.float)

        output = model(d1_smiles, d2_smiles, cl_embed, exprs)
        loss = loss_func(output, syn_score)

        opt.zero_grad()
        loss.backward()
        opt.step()

        loss_sum += loss.item()

    return loss_sum / dl_len


def val_epoch(model, loss_func, dl, device):

    loss_sum = 0.0
    dl_len = len(dl.dataset)

    true_vals = []
    pred_vals = []

    for batch in dl:

        cl_id, d1_id, d2_id, cl_embed, exprs, syn_score, d1_smiles, d2_smiles = batch

        d1_smiles = d1_smiles.to(device)
        d2_smiles = d2_smiles.to(device)
        cl_embed = cl_embed.to(device)
        exprs = exprs.to(device)

        syn_score = syn_score.to(device, dtype=torch.float)

        output = model(d1_smiles, d2_smiles, cl_embed, exprs)
        loss = loss_func(output, syn_score)
        loss_sum += loss.item()

        pred_vals.append(output.to("cpu"))
        true_vals.append(syn_score.to("cpu"))

    true_vals = torch.cat(true_vals)
    pred_vals = torch.cat(pred_vals)

    syn_pcc = pearson(pred_vals, true_vals)

    return loss_sum / dl_len, syn_pcc


# function to start training
def train(params, file_paths):

    num_epochs = params['num_epochs']
    device = params["device"]
    learning_rate = params["learning_rate"]

    log = open(params["log_fp"], "wt")
    log.write("#fold\tepoch\ttrain_loss\tval_loss\tpcc\n")

    model = Regressor()
    model.to(device)

    loss_func = nn.MSELoss(reduction="sum")
    opt = optim.Adam(model.parameters(), lr=learning_rate)
    best_loss = float('inf')
    best_model_state_dict = None

    k_folds = file_paths["database"]

    dataset = D2Dataset(k_folds, file_paths)
    ds = random_split(dataset, [0.9, 0.1])

    train_l = DataLoader(ds[0], batch_size=32, shuffle=True, num_workers=0)
    val_l = DataLoader(ds[1], batch_size=32, shuffle=True, num_workers=0)

    for epoch in tqdm.tqdm(range(num_epochs), desc="train epoch"):
        print('Epoch {}/{}, current lr={}'.format(epoch, num_epochs - 1, learning_rate))

        model.train()
        train_mse = train_epoch(model, loss_func, train_l, device, opt)

        model.eval()
        with torch.no_grad():
            val_mse, val_pcc = val_epoch(model, loss_func, val_l, device)

        print("current loss: (train) %s (val) %s  pcc: %s" % (train_mse, val_mse, val_pcc))

        log.write("%s\t%s\t%s\t%s\n" % (epoch, train_mse, val_mse, val_pcc))
        log.flush()

        if val_mse < best_loss:
            best_loss = val_mse
            best_model_state_dict = model.state_dict()

    model.load_state_dict(best_model_state_dict)
    log.close()

    return model, val_l


def infer(model, dl, params):

    device = params["device"]
    outfp = params["infer_fp"]

    outfile = open(outfp, "wt")

    for batch in dl:
        cl_id, d1_id, d2_id, cl_embed, exprs, syn_score, d1_smiles, d2_smiles = batch

        d1_smiles = d1_smiles.to(device)
        d2_smiles = d2_smiles.to(device)
        cl_embed = cl_embed.to(device)
        exprs = exprs.to(device)

        output = model(d1_smiles, d2_smiles, cl_embed, exprs)
        output = output.to("cpu")

        for cl_ii, d1_ii, d2_ii, o_ii, ss_ii in zip(cl_id, d1_id, d2_id, output, syn_score):
            outfile.write("%s\t%s\t%s\t%s\t%s\n" %
                          (cl_ii, d1_ii.item(), d2_ii.item(), o_ii.item(), ss_ii.item()))

    outfile.flush()
    outfile.close()



def main():
    file_paths = {
        "embeds": "../results/lwfd_m2_d1024_embed_0.pt",
        "exprs": "../data/parsed_24q4_expression.tsv",
        "smiles": "../data/downstream/ChemBERTa_embedded_smiles.pkl",
        "database": "../data/downstream/drugcomb_marsy_criteria_24q4.tsv"
    }

    local_time = datetime.now().strftime("%Y%M%d_%H:%M:%S")

    params = {
        "num_epochs": 100,
        "device": torch.device("cuda:0"),
        "learning_rate": 0.0001,
        "log_fp": "../%s_2drugs.log" % local_time,
        "infer_fp": "%s_inferred.txt" % local_time
    }

    model, dl = train(params, file_paths)
    infer(model, dl, params)

main()
