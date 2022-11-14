#! python3
# coding: utf-8

from conllu import parse
from ner_eval import Evaluator
from smart_open import open


def f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    score = 2 * (precision * recall) / (precision + recall)
    return score


def evaluation(predictions, gold):

    predictions = parse(open(predictions, "r").read())
    gold = parse(open(gold, "r").read())

    print(f"Gold sentences: {len(gold)}")

    gold_labels = []
    for sentence in gold:
        sentence = [token["misc"]["name"] for token in sentence]
        gold_labels.append(sentence)

    predicted_labels = []
    for sentence in predictions:
        sentence = [token["misc"]["name"] for token in sentence]
        predicted_labels.append(sentence)

    entities = ["PER", "ORG", "LOC", "GPE_LOC", "GPE_ORG", "PROD", "EVT", "DRV"]

    evaluator = Evaluator(gold_labels, predicted_labels, entities)
    results, results_agg = evaluator.evaluate()

    print("F1 scores:")
    for entity in results_agg:
        prec = results_agg[entity]["strict"]["precision"]
        rec = results_agg[entity]["strict"]["recall"]
        print(f"{entity}:\t{f1(prec, rec):.4f}")
    prec = results["strict"]["precision"]
    rec = results["strict"]["recall"]
    print(f"Overall score: {f1(prec, rec):.4f}")
    return f1(prec, rec)
