import pandas as pd
import torch
from torch.utils.data import DataLoader
import numpy as np
from models import RecSysGNN, AutoRec
from metrics import metrics as compute_metrics
from dataset import CustomDataset

N_USERS, N_ITEMS = 943, 1655

# K should be the same in all
K = 10


def baseline_relevance(user_weights, item_weights):
    relevance = torch.matmul(user_weights, torch.transpose(item_weights, 0, 1))
    return relevance


def eval_lightgcn(evaluate_df, adjacency):
    latent_dim = 64
    n_layers = 3

    lightgcn = RecSysGNN(
        latent_dim=latent_dim,
        num_layers=n_layers,
        num_users=N_USERS,
        num_items=N_ITEMS,
        model="LightGCN",
    )

    lightgcn.load_state_dict(torch.load("models/lightgcn.pt"))

    _, out = lightgcn(adjacency)
    final_user_Embed, final_item_Embed = torch.split(out, (N_USERS, N_ITEMS))

    relevance = baseline_relevance(final_user_Embed, final_item_Embed)
    return np.array(compute_metrics(evaluate_df, K, relevance))


def eval_ngcf(evaluate_df, adjacency):
    latent_dim = 64
    n_layers = 3

    ngcf = RecSysGNN(
        latent_dim=latent_dim,
        num_layers=n_layers,
        num_users=N_USERS,
        num_items=N_ITEMS,
        model="NGCF",
    )

    ngcf.load_state_dict(torch.load("models/ngcf.pt"))

    _, out = ngcf(adjacency)
    final_user_Embed, final_item_Embed = torch.split(out, (N_USERS, N_ITEMS))

    relevance = baseline_relevance(final_user_Embed, final_item_Embed)
    return np.array(compute_metrics(evaluate_df, K, relevance))


def eval_cdae(evaluate_df):
    autorec = AutoRec(N_USERS, N_ITEMS)

    autorec.load_state_dict(torch.load("models/autorec.pt"))

    dataset = CustomDataset(evaluate_df, N_USERS, N_ITEMS)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        preds = np.zeros_like(dataset.matrix)

        for idx, input_mat in dataloader:
            input_mat = input_mat.float()
            batch_pred = torch.sigmoid(autorec.forward(idx, input_mat))
            batch_pred = batch_pred.masked_fill(input_mat.bool(), float("-inf"))

            indices = idx.detach().cpu().numpy()
            preds[indices] = batch_pred.detach().cpu().numpy()

    preds = torch.from_numpy(preds)
    return np.array(compute_metrics(evaluate_df, K, preds))


if __name__ == "__main__":
    # load the data
    evaluate_df = pd.read_csv("benchmark/data/test.csv")
    test_adjacency = torch.stack(
        (
            torch.LongTensor(evaluate_df["user_id_idx"].values),
            torch.LongTensor(evaluate_df["item_id_idx"].values + N_USERS),
        )
    )

    print("Order of metrics: precision, recall, f1")

    with np.printoptions(precision=4, suppress=True):
        print(f"Metrics for LightGCN: {eval_lightgcn(evaluate_df, test_adjacency)}")
        print(f"Metrics for NGCF: {eval_ngcf(evaluate_df, test_adjacency)}")
        print(f"Metrics for CDAE: {eval_cdae(evaluate_df)}")
