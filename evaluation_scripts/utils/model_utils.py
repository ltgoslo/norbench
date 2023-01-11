#!/bin/env python3

import tensorflow as tf
import sentencepiece
import re
import git
from git.repo.base import Repo
import importlib
from transformers import (TFBertForSequenceClassification, BertTokenizer, AutoTokenizer, AutoModelForTokenClassification, BertTokenizerFast,
                          TFXLMRobertaForSequenceClassification, XLMRobertaTokenizer, TFAutoModelForSequenceClassification,
                          TFBertForTokenClassification, TFXLMRobertaForTokenClassification, TFAutoModelForTokenClassification,
                          TFDistilBertForTokenClassification, TFDistilBertForSequenceClassification, DistilBertTokenizer, 
                          DistilBertForTokenClassification, DistilBertTokenizerFast, BertForTokenClassification, XLMRobertaForTokenClassification)

from data_preparation.data_preparation_pos import *


models = {
    "bert": {
        "pos": TFBertForTokenClassification.from_pretrained,
        'ner': BertForTokenClassification.from_pretrained,
        "sentiment": TFBertForSequenceClassification.from_pretrained,
    },
    "xlm-roberta": {
        "pos": TFXLMRobertaForTokenClassification.from_pretrained,
        'ner': XLMRobertaForTokenClassification.from_pretrained,
        "sentiment": TFXLMRobertaForSequenceClassification.from_pretrained,
    },
    "distilbert": {
        "pos": TFDistilBertForTokenClassification.from_pretrained,
        'ner': DistilBertForTokenClassification.from_pretrained,
        "sentiment": TFDistilBertForSequenceClassification.from_pretrained,
    },
    "auto": {
        "pos": TFAutoModelForTokenClassification.from_pretrained,
        'ner': AutoModelForTokenClassification.from_pretrained,
        "sentiment": TFAutoModelForSequenceClassification.from_pretrained
    }
}

tokenizers = {
    "bert": {
        "pos": tokenizer_class_subword_tokenization(BertTokenizer).from_pretrained,
        'ner': BertTokenizerFast.from_pretrained,
        "sentiment": BertTokenizer.from_pretrained
    },
    "xlm-roberta": {
        "pos": tokenizer_class_subword_tokenization(XLMRobertaTokenizerFast).from_pretrained,
        'ner': XLMRobertaTokenizerFast.from_pretrained,
        "sentiment": XLMRobertaTokenizer.from_pretrained
    },
    "distilbert": {
        "pos": tokenizer_class_subword_tokenization(DistilBertTokenizer).from_pretrained,
        'ner': DistilBertTokenizerFast.from_pretrained,
        "sentiment": DistilBertTokenizer.from_pretrained
    },
    "auto": {
        "pos": tokenizer_class_subword_tokenization(AutoTokenizer).from_pretrained,
        'ner': AutoTokenizer.from_pretrained,
        "sentiment": AutoTokenizer.from_pretrained
    }
}

# new models can be added to the current dictionary
# or even if model is not added to the dictionary it can be still used  
model_names = {
        "bert-base-multilingual-cased": "bert-base-multilingual-cased",
        "tf-xlm-roberta-base": "jplu/tf-xlm-roberta-base",
        "mbert": "bert-base-multilingual-cased",
        "norbert1": "ltgoslo/norbert",
        "norbert": "ltgoslo/norbert",
        "norbert2": "ltgoslo/norbert2",
        "xlm-roberta-base": "xlm-roberta-base",
        "xlm-roberta": "xlm-roberta-base",
        "nb-bert-base": "NbAiLab/nb-bert-base",
        "distilbert": "distilbert-base-uncased",
        "scandibert": "vesteinn/ScandiBERT"
    }

paths_to_relevant_data = {
    'sentiment': ["https://github.com/ltgoslo/norec_sentence"],
    'pos': [
        'https://github.com/UniversalDependencies/UD_Norwegian-Bokmaal',
        'https://github.com/UniversalDependencies/UD_Norwegian-Nynorsk',
        'https://github.com/UniversalDependencies/UD_Norwegian-NynorskLIA'
        ],
    'ner': ['https://github.com/ltgoslo/norne']
}

def set_tf_memory_growth():
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu_devices[0], True)


def get_full_model_names(short_model_name, d=model_names):
    if short_model_name in d.values():
        return d[short_model_name] if short_model_name in d else short_model_name
    else:
        if short_model_name in d:
          model_name = d[short_model_name]
          return d[model_name] if model_name in d else model_name
        else:
          return short_model_name


def get_name_from_dict_keys(short_model_name, d=model_names):
    
    if short_model_name not in d and short_model_name not in d.values():
      short_name = short_model_name

    if short_model_name in d:
        short_name = short_model_name
    elif short_model_name in d.values():
        reverse_lookup = dict([(val, key) for key, val in d.items()])
        short_name = reverse_lookup[short_model_name]
    
    if 'roberta' in short_name or 'scandibert' in short_name.lower():
      return "xlm-roberta"
    elif 'distilbert' in short_name:
      return 'distilbert'
    elif 'bert' in short_name:
      return 'bert'
    else:
      return 'auto'


def get_relevant_auto_tokenizer_func(model):

    mod_class = model.__class__
    get_mod = re.search(r'(transformers\.models\.[Aa-zZ|_]+)\.', str(mod_class), re.DOTALL)
    get_mod_for_tokenizer = re.search(r'TF(.+)ForTokenClassification', str(mod_class), re.DOTALL)
    attr_list_for_model = dir(importlib.import_module(get_mod.group(1)))

    if get_mod_for_tokenizer != None:
      tokenizers = [tok for tok in attr_list_for_model if 'Tokenizer' in tok and get_mod_for_tokenizer.group(1) in tok]
      return getattr(importlib.import_module(get_mod.group(0)[:-1]), tokenizers[0])
    
    else:
      tokenizers = [tok for tok in attr_list_for_model if 'Tokenizer' in tok]
      return getattr(importlib.import_module(get_mod.group(0)[:-1]), tokenizers[0])


def create_model(short_model_name, task, num_labels, do_lower_case=False, from_pt=False):
    short_name = get_name_from_dict_keys(short_model_name)

    try:
        model = models[short_name][task](get_full_model_names(short_model_name), num_labels=num_labels)
    except:
        try:
            model = models[short_name][task](get_full_model_names(short_model_name), num_labels=num_labels, from_pt = True)
        except:
            model = models[short_name][task](get_full_model_names(short_model_name), num_labels=num_labels, from_tf=True)
    
    if short_name == "auto" and task == "pos":
      tokenizer_class = get_relevant_auto_tokenizer_func(model)
      try:
        tokenizer = tokenizer_class_subword_tokenization(tokenizer_class).from_pretrained(get_full_model_names(short_model_name))
      except:
        tokenizer = tokenizer_class_subword_tokenization(tokenizer_class).from_pretrained(get_full_model_names(short_model_name), do_lower_case)
    else:
      tokenizer = get_tokenizer(short_model_name, short_name, task)

    return model, tokenizer


def get_tokenizer(short_model_name, short_name, task, do_lower_case=False):
    if tokenizers[short_name][task] is not None:
        try:
            return tokenizers[short_name][task](get_full_model_names(short_model_name))
        except:
            return tokenizers[short_name][task](get_full_model_names(short_model_name), do_lower_case)
    else:
        try:
            return tokenizers['auto'][task](get_full_model_names(short_model_name))
        except:
            return tokenizers['auto'][task](get_full_model_names(short_model_name), do_lower_case)


def compile_model(model, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss)        
    return model


def make_batches(dataset, batch_size, repetitions, shuffle=True):
    if shuffle:
        dataset = dataset.shuffle(int(1e6), reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)
    n_batches = len(list(dataset.as_numpy_iterator()))
    dataset = dataset.repeat(repetitions)
    return dataset, n_batches


def download_datasets(task, language='no'):
    for paths in paths_to_relevant_data[task]:
        if len(paths_to_relevant_data[task]) > 1:
            sub_folder = paths.split('-')[1]
            Repo.clone_from(paths, f"data/{task}/{sub_folder}/")
        else:
            Repo.clone_from(paths, f"data/{task}/")
    print('cnoning was done')
