#!/bin/env python3

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.backend as backend


def filter_padding_tokens(test_examples, preds, label_map, tokenizer):
    """Filters padding tokens, labels, predictions and logits,
    then returns these as flattened lists, along with subword locations"""
    filtered_preds = []
    labels = []
    tokens = []
    logits = []
    subword_locations = []
    init = 0

    for i in range(len(test_examples)):
        example_tokens, example_labels, example_idx_map = tokenizer.subword_tokenize(
            test_examples[i]["tokens"], test_examples[i]["tags"])
        example_labels = [label_map[label] for label in example_labels]
        example_preds = preds[0][i, :len(example_labels)].argmax(axis=-1)
        example_logits = preds[0][i, :len(example_labels)]
        filtered_preds.extend(example_preds)
        labels.extend(example_labels)
        tokens.extend(example_tokens)
        logits.extend(example_logits)

        # Subwords
        counts = pd.Series(example_idx_map).value_counts(sort=False)
        example_idx_map = np.array(example_idx_map)
        all_indexes = np.arange(init, init + len(example_idx_map))
        for idx in counts[counts > 1].index:
            word_locs = all_indexes[example_idx_map == idx]
            subword_locations.append((word_locs[0], word_locs[-1] + 1))
        init += len(example_tokens)

    return tokens, labels, filtered_preds, logits, subword_locations


def find_subword_locations(tokens):
    """Finds the starting and ending index of words that have been broken into subwords"""
    subword_locations = []
    start = None
    for i in range(len(tokens)):
        if tokens[i].startswith("##") and not (tokens[i - 1].startswith("##")):
            start = i - 1
        if not (tokens[i].startswith("##")) and tokens[i - 1].startswith("##") and i != 0:
            end = i
            subword_locations.append((start, end))

    return subword_locations


def reconstruct_subwords(subword_locations, filtered_preds, logits):
    """Assemble subwords back into the original word in the global lists
    of tokens, labels and predictions, and select a predicted tag"""
    new_preds = []
    prev_end = 0

    for start, end in subword_locations:
        if len(set(filtered_preds[start:end])) > 1:
            # Subword predictions do not all agree
            temp = np.array([(M.max(), M.argmax()) for M in logits[start:end]])
            prediction = temp[temp[:, 0].argmax(), 1]
        else:
            prediction = filtered_preds[start]
        new_preds += filtered_preds[prev_end:start] + [prediction]
        prev_end = end

    # Last subword onwards
    new_preds += filtered_preds[prev_end:]

    return new_preds


def ignore_acc(y_true_class, y_pred_class, class_to_ignore=0):
    y_pred_class = backend.cast(backend.argmax(y_pred_class, axis=-1), 'int32')
    y_true_class = backend.cast(y_true_class, 'int32')
    ignore_mask = backend.cast(backend.not_equal(y_true_class, class_to_ignore), 'int32')
    matches = backend.cast(backend.equal(y_true_class, y_pred_class), 'int32') * ignore_mask
    accuracy = backend.sum(matches) / backend.maximum(backend.sum(ignore_mask), 1)
    return accuracy


def get_ud_tags():
    return ["O", "_", "ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM",
            "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]


def token_type_model_attr(model, max_length):
    # if model has token_type_ids as an attribute    
    try:
      input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
      attention_mask=tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")
      token_type_ids=tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name="token_type_ids")
      model_attr = model({'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids})
      return True
    except:
      return False


def entities_tokens():
    
    list_of_tags = get_ud_tags()
    marked_list_of_tags = [f"<{tag}>" for tag in list_of_tags]

    labels2words = {
    "O": '[Other]',
    "_": '[None]', 
    "ADJ": '[Adjective]',
    "ADP": '[Adposition]',
    "ADV": '[Adverb]',
    "AUX": '[Auxiliary]',
    "CCONJ": '[Coordconj]',
    "DET": '[Determiner]',
    "INTJ": '[Interjection]',
    "NOUN": '[Noun]',
    "NUM": '[Numeral]',
    "PART": '[Particle]',
    "PRON": '[Pronoun]',
    "PROPN": '[Propernoun]',
    "PUNCT": '[Punctuation]',
    "SCONJ": '[Subordconj]',
    "SYM": '[Symbol]',
    "VERB": '[Verb]',
    "X": '[Unseen]'}

    entities2markedtags = {v: f"<{k}>" for k, v in labels2words.items()}
    
    return list_of_tags, marked_list_of_tags, labels2words, entities2markedtags


def organized_subsets_t5(data, id2label):
    ids, tokens, tags, tags_indexed = [], [], [], []
    for el in data:
        ids.append(el['id'])
        tokens.append(el['tokens'])
        tags.append(el['tags'])
        tags_indexed.append([id2label[tag] for tag in el['tags']])
    return ids, tokens, tags, tags_indexed