#! /bin/env python3
# coding: utf-8

import pandas as pd
import sys

data = pd.read_csv(sys.argv[1], delimiter="\t", header=None)

data.columns = ["Time", "Seed", "Precision", "Recall", "F1"]

print(data)

print(f"Average F1 score: {data.F1.mean() * 100:.1f}")
print(f"Standard deviation: {data.F1.std() * 100:.1f}")
