#!/usr/bin/env python
# coding: utf-8

import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd
import numpy as np


def getdata(modeltype, task, lr):
    north_model = pd.read_csv(
        f"North5-{modeltype}_{lr}_{task}_validation.tsv", sep="\t"
    )
    ltg_model = pd.read_csv(f"nort5-{modeltype}_{lr}_{task}_validation.tsv", sep="\t")
    north = {"train_losses": [], "dev_f1s": [], "name": f"North5_{modeltype}_{lr}"}
    ltg = {"train_losses": [], "dev_f1s": [], "name": f"nort5_{modeltype}_{lr}"}

    for epoch in epochs:
        data_north = north_model.loc[north_model["epoch"] == epoch]
        north["dev_f1s"].append(data_north.dev_f1.mean())
        north["train_losses"].append(data_north.train_loss.mean())

        data_ltg = ltg_model.loc[ltg_model["epoch"] == epoch]
        ltg["dev_f1s"].append(data_ltg.dev_f1.mean())
        ltg["train_losses"].append(data_ltg.train_loss.mean())
    return north, ltg


modeltype = sys.argv[1]
task = sys.argv[2]

epoch_nr = 20

epochs = list(range(epoch_nr))

# lrs = ["1e3", "1e4", "1e5"]
lrs = ["3e4", "5e4", "7e4"]
# lrs = ["1e4", "2e4", "4e4", "6e4"]

colors = iter(["green", "red", "blue", "black"])

plt.style.use("ggplot")

plt.xlabel("Epochs")
plt.ylabel("Validation F1 scores")
plt.xticks(range(epoch_nr))
plt.xticks(fontsize=6, rotation=45)
plt.title(f"Sentiment classification for {modeltype} models ({task})")

for lr in lrs:
    north, ltg = getdata(modeltype, task, lr)
    color = next(colors)
    for model in [north, ltg]:
        if "nort5" in model["name"]:
            marker = "o"
        else:
            marker = "x"
        plt.plot(
            epochs,
            model["dev_f1s"],
            label=f"{model['name']}",
            color=color,
            marker=marker,
        )
        # plt.plot(epochs, model["train_losses"], label=f"{model['name']} train loss", linestyle='dashed')

plt.legend(loc="best")
plt.savefig(f"{modeltype}_{task}.png", dpi=300, bbox_inches="tight")
