import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, df, num_users, num_items):
        i = torch.stack(
            (
                torch.LongTensor(df["user_id_idx"].values),
                torch.LongTensor(df["item_id_idx"].values),
            )
        )

        # create a sparse tensor of shape (users, items) with ones at the indices
        # of the user-item pairs
        self.matrix = torch.sparse_coo_tensor(
            indices=i,
            values=torch.ones(len(df)),
            size=(num_users, num_items),
        ).to_dense()

        self.num_users = num_users
        self.num_items = num_items

    def __len__(self):
        return len(self.matrix)

    def __getitem__(self, idx):
        return idx, self.matrix[idx]
