import argparse
import random
import csv
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import umap
import umap.plot
from importlib.metadata import version
import torch

from model import DualEncoder, SingleEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--encoder_type", choices=["sbert", "luar"], default="sbert")
args = parser.parse_args()

#data_files = ["gpt4o_small", "llama70b_small", "gemini_small", "qwen32b_small", "qwen72b_small", "sonnet_small"]
data_files = ["gpt4o", "llama70b", "gemini", "qwen32b", "qwen72b", "sonnet", "llama8b"]
embedded_data = []
labels = []

for data_file_index, data_file in enumerate(data_files):
    data = pd.read_csv(data_file + ".csv")
    data["label"] = 0

    MAX_XJ_LENGTH = 30

    encoder = SingleEncoder(MAX_XJ_LENGTH, 1, args.encoder_type)

    if args.encoder_type == "sbert":    
        for x in data[data_file]:
            embedded_data.append(encoder.model.encode(x))
            labels.append(data_file)
    else:
        for x in data[data_file]:
            input_ids = encoder.tokenizer(x, return_tensors="pt", padding=True, truncation=True).to(encoder.model.device)
            input_ids['input_ids'] = input_ids['input_ids'].unsqueeze(1)
            input_ids['attention_mask'] = input_ids['attention_mask'].unsqueeze(1)
            
            output = encoder.model(**input_ids)
            embedded_data.append(output.detach().cpu().numpy())
            labels.append(data_file)

embedded_data = np.vstack(embedded_data)
labels = np.array(labels)

umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
embedding = umap_model.fit(embedded_data)  # Fit the UMAP model

umap.plot.points(embedding, labels=labels)
plt.savefig(f"umap_plot_{args.encoder_type}.png")