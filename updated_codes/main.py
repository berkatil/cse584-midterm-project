import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset import ModelDetectionDataset
from model import DualEncoder, SingleEncoder
import warnings
import os


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

train, val_test = train_test_split(data, test_size=0.3, random_state=42, shuffle=True)
val, test = train_test_split(val_test, test_size=0.33, random_state=42, shuffle=True)

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
    model = DualEncoder(MAX_XI_LENGTH, MAX_XJ_LENGTH, 1, args.encoder_type)
elif args.architecture_type == "single":
    model = SingleEncoder(MAX_XI_LENGTH + MAX_XJ_LENGTH, 1, args.encoder_type)
else:
    raise ValueError("Invalid architecture type")

# Move model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

best_val_loss = float("inf")
for epoch in range(1):
    model.train()
    for i, (xi, xj, labels) in enumerate(train_loader):
        labels = labels.to(device)  # Move labels to device
        optimizer.zero_grad()
        output = model(xi, xj)  # xi and xj are text and will be processed inside the model
        loss = loss_function(output, labels)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, batch {i}, loss {loss.item()}")

    model.eval()
    with torch.no_grad():
        val_losses = []
        for i, (xi, xj, labels) in enumerate(val_loader):
            labels = labels.to(device)  # Move labels to device
            output = model(xi, xj)
            loss = loss_function(output, labels)
            val_losses.append(loss.item())
        val_loss = np.mean(val_losses)
        print(f"Epoch {epoch}, val loss {val_loss}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

model.load_state_dict(torch.load("best_model.pth"))
model.eval()
with torch.no_grad():
    predictions = []
    ground_truths = []
    for i, (xi, xj, labels) in enumerate(test_loader):
        labels = labels.to(device)  # Move labels to device
        output = model(xi, xj)
        predictions.extend(output.detach().cpu().numpy().argmax(axis=1).tolist())
        ground_truths.extend(labels.detach().cpu().numpy().tolist())

prec, rec, f1, _ = precision_recall_fscore_support(ground_truths, predictions, average="macro")
print("Macro Average Scores")
print(f"Precision: {prec}, Recall: {rec}, F1: {f1}")
prec, rec, f1, _ = precision_recall_fscore_support(ground_truths, predictions, average="micro")
print("Micro Average Scores")
print(f"Precision: {prec}, Recall: {rec}, F1: {f1}")
prec, rec, f1, _ = precision_recall_fscore_support(ground_truths, predictions, average=None)
print("No Average Scores")
print(f"Precision: {prec}, Recall: {rec}, F1: {f1}")
accuracy = accuracy_score(ground_truths, predictions)
print(f"Accuracy: {accuracy}")
matrix = confusion_matrix(ground_truths, predictions)
print("Accuracy for each class")
print(matrix.diagonal()/matrix.sum(axis=1))

disp = ConfusionMatrixDisplay(confusion_matrix=matrix)

print("Confusion Matrix")
disp.plot()
plt.savefig('confusion_matrix_plot.png')
plt.show()