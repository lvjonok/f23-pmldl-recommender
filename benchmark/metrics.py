"""metrics module provides function computing precision and recall"""
import torch
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def metrics(test_df, K, interactions):
    """
    metrics function returns precision@K, recall@K, F1@K for a given test dataframe

    interactions - is a matrix of (n_users, n_items) shape with [0, 1] values with our predictions
    """

    # store top K predictions for each user
    top_k = torch.topk(interactions, K, dim=1).indices
    # move to numpy
    top_k = top_k.cpu().numpy()

    # store relevant items for each user from test dataframe
    relevant_items = (
        test_df.groupby("user_id_idx").apply(lambda x: x.item_id_idx.values).values
    )

    # compute relevant in top K value
    relevant_in_top_k = np.array(
        [
            len(np.intersect1d(top_k[i], relevant_items[i]))
            for i in range(len(relevant_items))
        ]
    )

    # compute recall@K = # relevant in top K / total # of relevant
    recall = np.mean(relevant_in_top_k / np.array([len(x) for x in relevant_items]))

    # compute precision@K = # relevant in top K / K
    precision = np.mean(relevant_in_top_k / K)

    # compute f1@K
    f1 = 2 * (precision * recall) / (precision + recall)

    return precision, recall, f1


def compute_relevance(user_Embed_wts, item_Embed_wts, train_df):
    """
    compute_relevance returns a matrix of shape (n_users, n_items)
    with values within [0, 1] which tells our prediction relevance for each user-item pair
    """
    n_users = user_Embed_wts.shape[0]
    n_items = item_Embed_wts.shape[0]

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
        torch.sparse_coo_tensor(i, v, (n_users, n_items)).to_dense().to(device)
    )

    # we want to consider relevance only for the test set items
    relevance_score = torch.mul(relevance_score, (1 - interactions_t))

    return relevance_score


# def get_metrics(
#     user_Embed_wts, item_Embed_wts, n_users, n_items, train_df, test_data, K
# ):
#     # torch.LongTensor(test_data["user_id_idx"].unique())
#     # compute the score of all user-item pairs
#     relevance_score = torch.matmul(
#         user_Embed_wts, torch.transpose(item_Embed_wts, 0, 1)
#     )

#     # create dense tensor of all user-item interactions
#     i = torch.stack(
#         (
#             torch.LongTensor(train_df["user_id_idx"].values),
#             torch.LongTensor(train_df["item_id_idx"].values),
#         )
#     )
#     v = torch.ones((len(train_df)), dtype=torch.float64)
#     interactions_t = (
#         torch.sparse.FloatTensor(i, v, (n_users, n_items)).to_dense().to(device)
#     )

#     # mask out training user-item interactions from metric computation
#     relevance_score = torch.mul(relevance_score, (1 - interactions_t))

#     # compute top scoring items for each user
#     topk_relevance_indices = torch.topk(relevance_score, K).indices
#     topk_relevance_indices_df = pd.DataFrame(
#         topk_relevance_indices.cpu().numpy(),
#         columns=["top_indx_" + str(x + 1) for x in range(K)],
#     )
#     topk_relevance_indices_df["user_ID"] = topk_relevance_indices_df.index
#     topk_relevance_indices_df["top_rlvnt_itm"] = topk_relevance_indices_df[
#         ["top_indx_" + str(x + 1) for x in range(K)]
#     ].values.tolist()
#     topk_relevance_indices_df = topk_relevance_indices_df[["user_ID", "top_rlvnt_itm"]]

#     # measure overlap between recommended (top-scoring) and held-out user-item
#     # interactions
#     test_interacted_items = (
#         test_data.groupby("user_id_idx")["item_id_idx"].apply(list).reset_index()
#     )
#     metrics_df = pd.merge(
#         test_interacted_items,
#         topk_relevance_indices_df,
#         how="left",
#         left_on="user_id_idx",
#         right_on=["user_ID"],
#     )
#     metrics_df["intrsctn_itm"] = [
#         list(set(a).intersection(b))
#         for a, b in zip(metrics_df.item_id_idx, metrics_df.top_rlvnt_itm)
#     ]

#     metrics_df["recall"] = metrics_df.apply(
#         lambda x: len(x["intrsctn_itm"]) / len(x["item_id_idx"]), axis=1
#     )
#     metrics_df["precision"] = metrics_df.apply(
#         lambda x: len(x["intrsctn_itm"]) / K, axis=1
#     )

#     return metrics_df["recall"].mean(), metrics_df["precision"].mean()
