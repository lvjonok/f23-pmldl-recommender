"""metrics module provides function computing precision and recall"""
# Standard library imports

# Third-party imports
import pandas as pd

pd.set_option("display.max_colwidth", None)

import torch

# from torch.utils.data import Dataset, DataLoader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_metrics(
    user_Embed_wts, item_Embed_wts, n_users, n_items, train_df, test_data, K
):
    test_user_ids = torch.LongTensor(test_data["user_id_idx"].unique())
    # compute the score of all user-item pairs
    relevance_score = torch.matmul(
        user_Embed_wts, torch.transpose(item_Embed_wts, 0, 1)
    )

    # create dense tensor of all user-item interactions
    i = torch.stack(
        (
            torch.LongTensor(train_df["user_id_idx"].values),
            torch.LongTensor(train_df["item_id_idx"].values),
        )
    )
    v = torch.ones((len(train_df)), dtype=torch.float64)
    interactions_t = (
        torch.sparse.FloatTensor(i, v, (n_users, n_items)).to_dense().to(device)
    )

    # mask out training user-item interactions from metric computation
    relevance_score = torch.mul(relevance_score, (1 - interactions_t))

    # compute top scoring items for each user
    topk_relevance_indices = torch.topk(relevance_score, K).indices
    topk_relevance_indices_df = pd.DataFrame(
        topk_relevance_indices.cpu().numpy(),
        columns=["top_indx_" + str(x + 1) for x in range(K)],
    )
    topk_relevance_indices_df["user_ID"] = topk_relevance_indices_df.index
    topk_relevance_indices_df["top_rlvnt_itm"] = topk_relevance_indices_df[
        ["top_indx_" + str(x + 1) for x in range(K)]
    ].values.tolist()
    topk_relevance_indices_df = topk_relevance_indices_df[["user_ID", "top_rlvnt_itm"]]

    # measure overlap between recommended (top-scoring) and held-out user-item
    # interactions
    test_interacted_items = (
        test_data.groupby("user_id_idx")["item_id_idx"].apply(list).reset_index()
    )
    metrics_df = pd.merge(
        test_interacted_items,
        topk_relevance_indices_df,
        how="left",
        left_on="user_id_idx",
        right_on=["user_ID"],
    )
    metrics_df["intrsctn_itm"] = [
        list(set(a).intersection(b))
        for a, b in zip(metrics_df.item_id_idx, metrics_df.top_rlvnt_itm)
    ]

    metrics_df["recall"] = metrics_df.apply(
        lambda x: len(x["intrsctn_itm"]) / len(x["item_id_idx"]), axis=1
    )
    metrics_df["precision"] = metrics_df.apply(
        lambda x: len(x["intrsctn_itm"]) / K, axis=1
    )

    return metrics_df["recall"].mean(), metrics_df["precision"].mean()
