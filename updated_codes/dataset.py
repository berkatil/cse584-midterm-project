import torch
from torch.utils.data import Dataset

class ModelDetectionDataset(Dataset):
    def __init__(self, data, concat=False):
        self.data = data
        self.concat = concat

    def __len__(self):
        return len(self.data)
        # returning comprehension by accessing data
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        xi = str(row["xi"])  # Ensure xi is a string
        xj = str(row["xj"])  # Ensure xj is a string
        label = torch.tensor(row["label"], dtype=torch.long)  # Ensure label is a tensor

        if xi in xj:
            xj = xj.replace(xi, "")
        if self.concat:
            return xi + xj, "", label

        return xi, xj, label