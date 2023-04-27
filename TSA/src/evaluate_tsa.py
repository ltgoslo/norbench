#! python3
# coding: utf-8


from src.ner_eval import Evaluator



def f1(precision, recall):
    if precision + recall == 0:
        return 0.0
    score = 2 * (precision * recall) / (precision + recall)
    return score


def evaluateur(gold, predictions):
    """Predictions and gold are lists of lists with token labels as text"""

    labels = set([l for s in gold for l in s])
    labels.remove('O') # remove 'O' label from evaluation
    labels  = list(set([l[2:] for l in labels]))
    sorted_labels = sorted(labels,key=lambda name: (name[1:], name[0]))

    evaluator = Evaluator(gold, predictions, sorted_labels)
    results, results_agg = evaluator.evaluate()

    # print("F1 scores:")
    for entity in results_agg:
        prec = results_agg[entity]["strict"]["precision"]
        rec = results_agg[entity]["strict"]["recall"]
        # print(f"{entity}:\t{f1(prec, rec):.4f}")
    prec = results["strict"]["precision"]
    rec = results["strict"]["recall"]
   # print(f"Overall score: {f1(prec, rec):.4f}")
    return f1(prec, rec)