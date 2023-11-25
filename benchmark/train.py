"""train module wraps the training and evaluation process"""
from benchmark.loss import bpr_loss as compute_brp_loss
from benchmark.metrics import get_metrics
from dataclasses import dataclass
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dataloader(df: pd.DataFrame, batch_size: int = 32):
    """
    dataloader uses BRP idea to create batches of data
    """

    n_usr = df.user_id_idx.nunique()
    n_itm = df.item_id_idx.nunique()

    def sample_neg(x):
        while True:
            neg_id = random.randint(0, n_itm - 1)
            if neg_id not in x:
                return neg_id

    interected_items_df = (
        df.groupby("user_id_idx")["item_id_idx"].apply(list).reset_index()
    )
    indices = [x for x in range(n_usr)]

    if n_usr < batch_size:
        users = [random.choice(indices) for _ in range(batch_size)]
    else:
        users = random.sample(indices, batch_size)
    users.sort()
    users_df = pd.DataFrame(users, columns=["users"])

    interected_items_df = pd.merge(
        interected_items_df,
        users_df,
        how="right",
        left_on="user_id_idx",
        right_on="users",
    )
    pos_items = (
        interected_items_df["item_id_idx"].apply(lambda x: random.choice(x)).values
    )
    neg_items = interected_items_df["item_id_idx"].apply(lambda x: sample_neg(x)).values

    return (
        torch.LongTensor(list(users)).to(device),
        torch.LongTensor(list(pos_items)).to(device) + n_usr,
        torch.LongTensor(list(neg_items)).to(device) + n_usr,
    )

    # # sample 'batch_size' users
    # users = np.random.choice(n_users, size=batch_size, replace=False)
    # # sort users
    # users.sort()

    # # helper function to sample from dataframe group
    # def sample(group):
    #     return group.sample(1)

    # # sample by one existing item for each user
    # items = (
    #     df[df.user_id_idx.isin(users)].groupby("user_id_idx").apply(sample).item_id_idx
    # )

    # # create temporary table where there will be all user-item non-existing pairs
    # df_outside_batch = df[~df.user_id_idx.isin(users)]

    # non_items = (
    #     df_outside_batch.groupby("user_id_idx")
    #     .apply(lambda x: np.random.choice(x.item_id_idx))
    #     .sample(batch_size)
    # )

    # # for each item we have to add the number of users so that the index will be unique among users
    # items = items + n_users
    # non_items = non_items + n_users

    # # return tensors
    # return (
    #     torch.Tensor(list(users)).long().to(device),
    #     torch.Tensor(list(items)).long().to(device),
    #     torch.Tensor(list(non_items)).long().to(device),
    # )


@dataclass
class TrainParameters:
    EPOCHS: int
    BATCH_SIZE: int
    DECAY: float
    LR: float
    K: int


def train_and_eval(
    model,
    optimizer,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    params: TrainParameters,
):
    loss_list_epoch = []
    bpr_loss_list_epoch = []
    reg_loss_list_epoch = []

    n_users = train_df.user_id_idx.nunique()
    n_items = train_df.item_id_idx.nunique()

    train_adjacency = torch.stack(
        (
            torch.LongTensor(train_df["user_id_idx"].values),
            torch.LongTensor(train_df["item_id_idx"].values + n_users),
        )
    ).to(device)

    recall_list = []
    precision_list = []

    for epoch in tqdm(range(params.EPOCHS)):
        n_batch = int(len(train_df) / params.BATCH_SIZE)

        final_loss_list = []
        bpr_loss_list = []
        reg_loss_list = []

        model.train()
        for batch_idx in range(n_batch):
            # print(batch_idx)
            optimizer.zero_grad()

            users, pos_items, neg_items = dataloader(train_df, params.BATCH_SIZE)
            (
                users_emb,
                pos_emb,
                neg_emb,
                userEmb0,
                posEmb0,
                negEmb0,
            ) = model.encode_minibatch(users, pos_items, neg_items, train_adjacency)

            bpr_loss, reg_loss = compute_brp_loss(
                users, users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0
            )
            reg_loss = params.DECAY * reg_loss
            final_loss = bpr_loss + reg_loss

            final_loss.backward()
            optimizer.step()

            final_loss_list.append(final_loss.item())
            bpr_loss_list.append(bpr_loss.item())
            reg_loss_list.append(reg_loss.item())

        model.eval()
        with torch.no_grad():
            _, out = model(train_adjacency)
            final_user_Embed, final_item_Embed = torch.split(out, (n_users, n_items))
            test_topK_recall, test_topK_precision = get_metrics(
                final_user_Embed,
                final_item_Embed,
                n_users,
                n_items,
                train_df,
                test_df,
                params.K,
            )

        loss_list_epoch.append(round(np.mean(final_loss_list), 4))
        bpr_loss_list_epoch.append(round(np.mean(bpr_loss_list), 4))
        reg_loss_list_epoch.append(round(np.mean(reg_loss_list), 4))

        recall_list.append(round(test_topK_recall, 4))
        precision_list.append(round(test_topK_precision, 4))

    return (
        loss_list_epoch,
        bpr_loss_list_epoch,
        reg_loss_list_epoch,
        recall_list,
        precision_list,
    )
