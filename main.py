import argparse
import random

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

from dataset import ModelDetectionDataset
from model import DualEncoder, SingleEncoder

def set_all_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument("--data_file", default="gpt4o.csv")
parser.add_argument("--architecture_type", choices=["dual", "single"], default="single")
parser.add_argument("--encoder_type", choices=["sbert", "luar"], default="sbert")

args = parser.parse_args()
set_all_seeds(42)
data = pd.read_csv(args.data_file)
data["label"] = 0

train, val_test = train_test_split(data, test_size=0.3, random_state=42)
val, test = train_test_split(val_test, test_size=0.33, random_state=42)

train_set = ModelDetectionDataset(train, args.architecture_type == "single")
val_set = ModelDetectionDataset(val, args.architecture_type == "single")
test_set = ModelDetectionDataset(test, args.architecture_type == "single")

loss_function = torch.nn.CrossEntropyLoss()

LR = 0.0001
BATCH_SIZE = 8
MAX_XI_LENGTH = 128
MAX_XJ_LENGTH = 30

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

if args.architecture_type == "dual":
    model = DualEncoder(MAX_XI_LENGTH, MAX_XJ_LENGTH, 1 ,args.encoder_type)
elif args.architecture_type == "single":
    model = SingleEncoder(MAX_XI_LENGTH + MAX_XJ_LENGTH, 1, args.encoder_type)
else:
   raise ValueError("Invalid architecture type")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_loss = float("inf")
for epoch in range(10):
    model.train()
    for i, (xi, xj, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(xi, xj)
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, batch {i}, loss {loss.item()}")

    model.eval()
    with torch.no_grad():
        val_losses = []
        for i, (xi, xj, labels) in enumerate(val_loader):
            output = model(xi, xj)
            loss = loss_function(output, labels)
            val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
        print(f"Epoch {epoch}, val loss {val_loss}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")