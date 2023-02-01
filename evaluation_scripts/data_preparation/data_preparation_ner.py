#!/bin/env python3

import utils.ner_utils as ner_utils
from utils.utils import read_conll
from transformers import DataCollatorForTokenClassification
import glob
import torch
import datasets
from datasets import load_metric, load_dataset, Dataset


tagset = ner_utils.get_ner_tags()


def load_dataset_ner(data_path, dataset_name="test"):
    """Loads conllu file, returns a list of dictionaries (one for each sentence) and a TF dataset"""
    try:
        data = read_conll(glob.glob(data_path + "/*{}.conllu".format(dataset_name.split("_")[0]))[0], label_nr=9)
    except:
        data = read_conll(glob.glob(data_path + "*{}.conllu".format(dataset_name.split("_")[0]))[0], label_nr=9)
    examples = [{"id": sent_id, "tokens": tokens, "ner_tags": [tag.split("|")[-1].split('=')[1] for tag in tags]} for sent_id, tokens, tags in
                zip(data[0], data[1], data[2])]

    return examples


def collecting_data(tokenizer, path, max_length=512, full_pipeline=True):
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

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer, padding='longest', max_length=max_length)
    tokenized_data = data.map(ner_utils.tokenize_and_align_labels, fn_kwargs={'tokenizer': tokenizer}, batched=True)

    return tokenized_data, data_collator


### Functions for T5 models

def tokenization_func(tokenizer, doc, max_len):
  return tokenizer.encode_plus(
                            doc, 
                            max_length=max_len,
                            truncation=True,
                            padding="max_length",
                            return_attention_mask=True,
                            return_token_type_ids=False,
                            return_tensors='pt'
                       )

class DatasetPreparstion(Dataset):

    def __init__(self, dataset, tokenizer, max_len, source_text, target_text):

        self.tokenizer = tokenizer
        self.dataset = dataset
        self.text_len = max_len
        self.source_text = source_text
        self.target_text = target_text
    
    def __len__(self):
        return len(self.dataset)


    def reorganize_target_text(self, index):

        list_of_curs = [(tag, self.source_text[index][idx], idx) for idx, tag in enumerate(self.target_text[index]) if tag != 'O']
        seq_cur_spans = ''
        prev_tag = ''

        for idx, cur_inf in enumerate(list_of_curs):
          orig_tag, tok, ind = cur_inf
          B_2E, spl_tags = orig_tag.split('-')[0], orig_tag.split('-')[1]
          if B_2E == 'B':
            seq_cur_spans += '{}: {}'.format(spl_tags, tok)
            if len(list_of_curs) <= idx + 1:
              pass
            elif list_of_curs[idx + 1][2] != ind + 1:
              seq_cur_spans += '; '
          

          if B_2E == 'I':
            if prev_tag == spl_tags and list_of_curs[idx - 1][2] == ind - 1:
              seq_cur_spans += ' ' + tok
            if len(list_of_curs) <= idx + 1:
              pass
            elif list_of_curs[idx + 1][2] != ind + 1:
              seq_cur_spans += '; '
          prev_tag = spl_tags
        return seq_cur_spans          
    
    
    def __getitem__(self, index):
        source_text = ' '.join(self.source_text[index]).lower() + ' </s>'
        target_text = self.reorganize_target_text(index).lower() + ' </s>'
        source = tokenization_func(self.tokenizer, source_text, self.text_len)
        target = tokenization_func(self.tokenizer, target_text, self.text_len)

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "target_mask": target_mask.to(dtype=torch.long),
        }


def collecting_data_t5(tokenizer, path, max_length=512, full_pipeline=True):
    "collecting data from the files"

    id2label = {t: i for i, t in enumerate(tagset)}

    if full_pipeline == True:
        train_data = load_dataset_ner(path, dataset_name='train')
        val_data = load_dataset_ner(path, dataset_name='dev')
        test_data = load_dataset_ner(path, dataset_name='test')

        tr_ids, tr_tokens, tr_tags, tr_tags_indexed  = ner_utils.organized_subset_t5(train_data, id2label)
        val_ids, val_tokens, val_tags, val_tags_indexed = ner_utils.organized_subset_t5(val_data, id2label)
        test_ids, test_tokens, test_tags, test_tags_indexed = ner_utils.organized_subset_t5(test_data, id2label)

        train_dataset_data = DatasetPreparstion(
            dataset=train_data,
            tokenizer=tokenizer,
            max_len=max_length,
            source_text=tr_tokens,
            target_text=tr_tags,
        )

        val_dataset_data = DatasetPreparstion(
            dataset=val_data,
            tokenizer=tokenizer,
            max_len=max_length,
            source_text=val_tokens,
            target_text=val_tags,
        )

        test_dataset_data = DatasetPreparstion(
            dataset=test_data,
            tokenizer=tokenizer,
            max_len=max_length,
            source_text=test_tokens,
            target_text=test_tags,
        )
        return train_dataset_data, val_dataset_data, test_dataset_data

    else:
        test_data = load_dataset_ner(path, dataset_name='test')
        test_ids, test_tokens, test_tags, test_tags_indexed = ner_utils.organized_subset_t5(test_data, id2label)
        test_dataset_data = DatasetPreparstion(
            dataset=test_data,
            tokenizer=tokenizer,
            max_len=max_length,
            source_text=test_tokens,
            target_text=test_tags,
        )

        return test_dataset_data
