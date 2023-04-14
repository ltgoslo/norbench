#!/bin/env python3

from transformers import BertTokenizer, XLMRobertaTokenizer, XLMRobertaTokenizerFast, DistilBertTokenizer, AutoTokenizer
from transformers.data.processors.utils import InputFeatures
import tensorflow as tf
import logging
import glob
import torch
from datasets import load_metric, load_dataset, Dataset
from utils.utils import read_conll
from utils.pos_utils import token_type_model_attr, organized_subsets_t5, entities_tokens


pos_tags, marked_tags, labels2words, entities2markedtags = entities_tokens()    
 

def tokenizer_class_subword_tokenization(value):
  class TTokenizer(value):
      def subword_tokenize(self, tokens, labels):
          # This propogates the label over any subwords that
          # are created by the byte-pair tokenization for training

          # IMPORTANT: For testing, you will have to undo this step by combining
          # the subword elements

          split_tokens, split_labels = [], []
          idx_map = []
          for ix, token in enumerate(tokens):
              sub_tokens = self.wordpiece_tokenizer.tokenize(token) if 'wordpiece_tokenizer' in dir(self) else self.tokenize(token)
              for jx, sub_token in enumerate(sub_tokens):
                  split_tokens.append(sub_token)
                  split_labels.append(labels[ix])
                  idx_map.append(ix)
          return split_tokens, split_labels, idx_map
  return TTokenizer


def convert_examples_to_tf_dataset(examples, tokenizer, model, tagset, max_length, token_type_ids_input=True):
    
    """
    token_type_ids_input: True for BERT as default, if model doesn't provide token_type_ids in attributes, False
    """
    
    features = []  # -> will hold InputFeatures to be converted later
    token_type_attr = token_type_model_attr(model, max_length)

    for e in examples:
        tokens = e["tokens"]
        labels = e["tags"]
        label_map = {label: i for i, label in enumerate(tagset)}  # Tags to indexes

        # Tokenize subwords and propagate labels
        split_tokens, split_labels, idx_map = tokenizer.subword_tokenize(tokens, labels)

        input_ids = tokenizer.convert_tokens_to_ids(split_tokens)
        attention_mask = [1] * len(input_ids)

        label_ids = [label_map[label] for label in split_labels]

        padding = [0] * (max_length - len(input_ids))
        input_ids += padding
        attention_mask += padding
        label_ids += padding
        token_type_ids = [0] * max_length


        # Create features
        if token_type_ids_input == True and token_type_attr == True:
            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                    label=label_ids
                )
            )
        else:
            features.append(
                InputFeatures(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    label=label_ids
                )
            )

    def gen():

        if token_type_ids_input == True and token_type_attr == True:
            for f in features:
                yield (
                    {
                        "input_ids": f.input_ids,
                        "attention_mask": f.attention_mask,
                        "token_type_ids": f.token_type_ids,
                    },
                    f.label,
                )
        else:
            for f in features:
                yield (
                {
                    "input_ids": f.input_ids,
                    "attention_mask": f.attention_mask,
                },
                f.label,
            )
    if token_type_ids_input == True and token_type_attr == True:
        return tf.data.Dataset.from_generator(
              gen,
              ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
              (
                  {
                      "input_ids": tf.TensorShape([None]),
                      "attention_mask": tf.TensorShape([None]),
                      "token_type_ids": tf.TensorShape([None]),
                  },
                  tf.TensorShape([None]),
              ),
          )
    else:
        return tf.data.Dataset.from_generator(
        gen,
        ({"input_ids": tf.int32, "attention_mask": tf.int32}, tf.int64),
        (
            {
                "input_ids": tf.TensorShape([None]),
                "attention_mask": tf.TensorShape([None]),
            },
            tf.TensorShape([None]),
        ),
      )


def load_dataset(data_path, tokenizer, model, max_length, tagset, dataset_name="test"):
    """Loads conllu file, returns a list of dictionaries (one for each sentence) and a TF dataset"""
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    
    try:
        data = read_conll(glob.glob(data_path + "*{}.conllu".format(dataset_name.split("_")[0]))[0]) 
    except:
        data = read_conll(glob.glob(data_path + "/*{}.conllu".format(dataset_name.split("_")[0]))[0])

    examples = [{"id": sent_id, "tokens": tokens, "tags": tags} for sent_id, tokens, tags in
                zip(data[0], data[1], data[2])]
    
    # In case some example is over max length
    examples = [example for example in examples if len(
            tokenizer.subword_tokenize(example["tokens"], example["tags"])[0]) <= max_length]


    token_type_ids_input = False if ('roberta' in tokenizer.name_or_path or 'distilbert' in tokenizer.name_or_path) else True 

    try:
        dataset = convert_examples_to_tf_dataset(examples=examples, tokenizer=tokenizer, model=model,
                                                            tagset=tagset, max_length=max_length, token_type_ids_input=token_type_ids_input)
    except:
        dataset = convert_examples_to_tf_dataset(examples=examples, tokenizer=tokenizer, model=model,
                                                        tagset=tagset, max_length=max_length, token_type_ids_input=False)


    return examples, dataset
    # This loops 3 times over the same data, including the convert to TF, could it be done in one?


### Functions for T5 models

def tokenization_func(tokenizer, doc, max_len):
  return tokenizer.encode_plus(
                            doc,  # document to encode.
                            add_special_tokens=True,  # add tokens relative to model
                            max_length=max_len,  # set max length
                            truncation=True,  # truncate longer messages
                            padding="max_length",  # add padding
                            return_attention_mask=True,  # create attn. masks
                            return_token_type_ids=False,
                            return_tensors='pt'  # return pytorch tensors
                       )


def load_dataset_t5(data_path, dataset_name="test"):
    """Loads conllu file, returns a list of dictionaries (one for each sentence) and a TF dataset"""
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    
    try:
        data = read_conll(glob.glob(data_path + "*{}.conllu".format(dataset_name.split("_")[0]))[0]) 
    except:
        data = read_conll(glob.glob(data_path + "/*{}.conllu".format(dataset_name.split("_")[0]))[0])

    examples = [{"id": sent_id, "tokens": tokens, "tags": tags} for sent_id, tokens, tags in
                zip(data[0], data[1], data[2])]
    
    return examples


### THE BEGGINING OF T5 MODEL IMPLEMENTATION

class DatasetPreparstion(Dataset):

    def __init__(
        self, dataset, tokenizer, max_len, source_text, target_text, prefix='Extract Entities:'
    ):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.text_len = max_len
        
        self.source_text = source_text
        self.target_text = target_text
        self.prefix = prefix

        self.upd_targets = self.reorganize_target_text()

    
    def __len__(self):
        return len(self.dataset)


    def reorganize_target_text(self):

        updated_tagets = []
        for text, labels in zip(self.source_text, self.target_text):
          upd_str = ''
          for token, lab in zip(text, labels):
            upd_str += f'{token} {labels2words[lab]} '
          updated_tagets.append(upd_str.strip())
        
        return updated_tagets


    def start_ignore_index(self, target_token_ids, value=-100):
        index = len(target_token_ids)
        if value in target_token_ids:
            index = target_token_ids.index(value)
        return index
    
    
    def __getitem__(self, index):

        source_text = ' '.join(self.source_text[index])
        source_text = f'{self.prefix} {source_text}'.strip() + self.tokenizer.eos_token
        target_text = str(self.upd_targets[index])


        target_text = str(self.upd_targets[index]) + self.tokenizer.eos_token

        source = tokenization_func(self.tokenizer, source_text, self.text_len)
        target = tokenization_func(self.tokenizer, target_text, self.text_len)

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze()
        target_mask = target["attention_mask"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
            "attention_mask": source_mask.to(dtype=torch.long),
            "ignore_ind": self.start_ignore_index(target_ids),
        }


def collecting_data_t5(tokenizer, path, max_length=512, full_pipeline=True):
    "collecting data from the files"

    pos_tags, marked_tags, labels2words, entities2markedtags = entities_tokens()    
    id2label = {t: i for i, t in enumerate(pos_tags)}

    if full_pipeline == True:
        train_data = load_dataset_t5(path, dataset_name='train')
        val_data = load_dataset_t5(path, dataset_name='dev')
        test_data = load_dataset_t5(path, dataset_name='test')

        tr_ids, tr_tokens, tr_tags, tr_tags_indexed = organized_subsets_t5(train_data, id2label)
        val_ids, val_tokens, val_tags, val_tags_indexed = organized_subsets_t5(val_data, id2label)
        test_ids, test_tokens, test_tags, test_tags_indexed = organized_subsets_t5(test_data, id2label)

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
        test_data = load_dataset_t5(path, dataset_name='test')
        test_ids, test_tokens, test_tags, test_tags_indexed = organized_subsets_t5(test_data, id2label)
        test_dataset_data = DatasetPreparstion(
            dataset=test_data,
            tokenizer=tokenizer,
            max_len=max_length,
            source_text=test_tokens,
            target_text=test_tags,
        )

        return test_dataset_data