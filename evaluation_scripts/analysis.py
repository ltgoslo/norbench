#! /bin/env python3
# coding: utf-8

import pandas as pd
import sys

data = pd.read_csv(sys.argv[1], delimiter="\t", header=None)

data.columns = ["Seed", "Dev_F1", "Test_F1"]
print(data)

mean_f1 = data.Test_F1.mean() * 100
std_f1 = data.Test_F1.std() * 100

mean_dev_f1 = data.Dev_F1.mean() * 100
std_dev_f1 = data.Dev_F1.std() * 100

print(f"Average test F1 score: {mean_f1:.1f}")
print(f"Standard deviation: {std_f1:.1f}")

data.loc[len(data.index)] = ["Mean", mean_dev_f1, mean_f1]
data.loc[len(data.index)] = ["Deviation", std_dev_f1, std_f1]

data.to_csv(sys.argv[1], sep="\t", index=False)
