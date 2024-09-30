from torch.utils.data import Dataset


class ModelDetectionDataset(Dataset):
    def __init__(self, data, concat=False):
        self.data = data
        self.concat = concat

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        if self.concat:
            return row["xi"] + row["xj"], "", row["label"]
        return row["xi"], row["xj"], row["label"]