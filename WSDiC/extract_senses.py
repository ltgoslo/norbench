#! /bin/env python3
#! coding: utf-8

import sys

words = {}

for line in sys.stdin:
    if not line.strip():
        continue
    res = line.strip().split("\t")
    if len(res) < 2:
        continue
    (lemma, sense) = res
    if not sense.strip():
        continue
    if not lemma in words:
        words[lemma] = {}
    if not sense in words[lemma]:
        words[lemma][sense] = 0
    words[lemma][sense] += 1

for w in words:
    if len(words[w]) > 1:
        out = True
        for sense in words[w]:
            if words[w][sense] < 2:
                out = False
        if out:
            print(w, words[w])
