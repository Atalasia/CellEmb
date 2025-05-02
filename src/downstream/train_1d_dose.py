from dataset_1d_dose import D1DoseDataset
from model_1d_dose import Regressor

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from datetime import datetime
import os
import tqdm


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

        cl_id, d_id, exprs, embeds, dose, ic50, d_smiles = batch

        d_smiles = d_smiles.to(device)
        embeds = embeds.to(device)
        exprs = exprs.to(device)
        dose = dose.to(device, dtype=torch.float)
        ic50 = ic50.to(device, dtype=torch.float)

        output = model(d_smiles, embeds, exprs, dose)
        loss = loss_func(output, ic50)

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

        cl_id, d_id, exprs, embeds, dose, ic50, d_smiles = batch

        d_smiles = d_smiles.to(device)
        embeds = embeds.to(device)
        exprs = exprs.to(device)
        dose = dose.to(device, dtype=torch.float)
        ic50 = ic50.to(device, dtype=torch.float)

        output = model(d_smiles, embeds, exprs, dose)
        loss = loss_func(output, ic50)

        loss_sum += loss.item()

        pred_vals.append(output.to("cpu"))
        true_vals.append(ic50.to("cpu"))

    true_vals = torch.cat(true_vals)
    pred_vals = torch.cat(pred_vals)

    ic50_pcc = pearson(pred_vals, true_vals)

    return loss_sum / dl_len, ic50_pcc


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
    dataset = D1DoseDataset(k_folds, file_paths)
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
        cl_id, d_id, exprs, embeds, dose, target, d_smiles = batch

        d_smiles = d_smiles.to(device)
        embeds = embeds.to(device)
        exprs = exprs.to(device)
        dose = dose.to(device, dtype=torch.float)
        target = target.to(device, dtype=torch.float)

        output = model(d_smiles, embeds, exprs, dose)
        output = output.to("cpu")

        for cl_ii, d_ii, dd_ii, o_ii, ss_ii in zip(cl_id, d_id, dose, output, target):
            outfile.write("%s\t%s\t%s\t%s\t%s\n" %
                          (cl_ii, d_ii.item(), dd_ii.item(), o_ii.item(), ss_ii.item()))

    outfile.flush()
    outfile.close()


def main():
    file_paths = {

        "smiles": "../data/downstream/ChemBERTa_embedded_smiles.pkl",
        "exprs": "../data/parsed_24q4_expression.tsv",
        "embeds": "../results/lwfd_m2_d1024_embed_0.pt",
        "database": "../data/downstream/gcsi_gr_1.3.txt"
    }

    local_time = datetime.now().strftime("%Y%M%d_%H:%M:%S")

    params = {
        "num_epochs": 100,
        "device": torch.device("cuda:0"),
        "learning_rate": 0.0001,
        "log_fp": "../%s_1drug_dosage.log" % local_time,
        "infer_fp": "%s_inferred.txt" % local_time
    }

    model, dl = train(params, file_paths)
    infer(model, dl, params)


main()
