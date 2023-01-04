#!/bin/env python3

import utils.ner_utils as ner_utils
from utils.utils import read_conll
from transformers import DataCollatorForTokenClassification
import glob
import datasets
from datasets import load_metric, load_dataset, Dataset


tagset = ner_utils.get_ner_tags()


def load_dataset_ner(data_path, dataset_name="test"):
    """Loads conllu file, returns a list of dictionaries (one for each sentence) and a TF dataset"""

    data = read_conll(glob.glob(data_path + "/*{}.conllu".format(dataset_name.split("_")[0]))[0], label_nr=9)
    examples = [{"id": sent_id, "tokens": tokens, "ner_tags": [tag.split("|")[-1].split('=')[1] for tag in tags]} for sent_id, tokens, tags in
                zip(data[0], data[1], data[2])]

    return examples


def collecting_data(tokenizer, path, full_pipeline=True):
    "collecting data from the files"

    id2label = {t: i for i, t in enumerate(tagset)}

    if full_pipeline == True:
        train_data = load_dataset_ner(path, dataset_name='train')
        dev_data = load_dataset_ner(path, dataset_name='dev')
        test_data = load_dataset_ner(path, dataset_name='test')

        tr_ids, tr_tokens, tr_tags = ner_utils.organized_subsets(train_data, id2label)
        de_ids, de_tokens, de_tags = ner_utils.organized_subsets(dev_data, id2label)
        te_ids, te_tokens, te_tags = ner_utils.organized_subsets(test_data, id2label)

        data = datasets.DatasetDict({'train': Dataset.from_dict({'id': tr_ids,'tokens': tr_tokens, 'tags': tr_tags}),
                                    'dev': Dataset.from_dict({'id': de_ids,'tokens': de_tokens, 'tags': de_tags}),
                                    'test': Dataset.from_dict({'id': te_ids,'tokens': te_tokens, 'tags': te_tags})})
    else:
        test_data = load_dataset_ner(path, dataset_name='test')
        te_ids, te_tokens, te_tags = ner_utils.organized_subsets(test_data, id2label)
        data = datasets.DatasetDict({'test': Dataset.from_dict({'id': te_ids,'tokens': te_tokens, 'tags': te_tags})})

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    tokenized_data = data.map(ner_utils.tokenize_and_align_labels, fn_kwargs={'tokenizer': tokenizer}, batched=True)

    return tokenized_data, data_collator
