"""loss module provides function computing BRP loss"""
import torch
import torch.nn.functional as F


def bpr_loss(users, users_emb, pos_emb, neg_emb, user_emb0, pos_emb0, neg_emb0):
    # compute loss from initial embeddings, used for regulization
    reg_loss = (
        (1 / 2)
        * (user_emb0.norm().pow(2) + pos_emb0.norm().pow(2) + neg_emb0.norm().pow(2))
        / float(len(users))
    )

    # compute BPR loss from user, positive item, and negative item embeddings
    pos_scores = torch.mul(users_emb, pos_emb).sum(dim=1)
    neg_scores = torch.mul(users_emb, neg_emb).sum(dim=1)

    bpr_loss = torch.mean(F.softplus(neg_scores - pos_scores))

    # brp_loss2 = torch.sum(torch.log(torch.sigmoid(pos_scores - neg_scores)))

    # return brp_loss2, reg_loss

    return bpr_loss, reg_loss
