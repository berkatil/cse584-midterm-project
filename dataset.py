from torch.utils.data import Dataset


class ModelDetectionDataset(Dataset):
    def __init__(self, data, concat=False):
        self.data = data
        self.concat = concat

    def __len__(self):
        return 5#len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        xi = row["xi"]
        xj = row["xj"]
        if xi in xj:
            xj = xj.replace(xi, "")
        if self.concat:
            return xi + xj, "", row["label"]
       
        return xi, xj, row["label"]