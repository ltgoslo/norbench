#!/bin/env python3

from transformers import BertTokenizer, XLMRobertaTokenizer, XLMRobertaTokenizerFast, DistilBertTokenizer, AutoTokenizer
from transformers.data.processors.utils import InputFeatures
import tensorflow as tf
import logging
import glob
from utils.utils import read_conll
from utils.pos_utils import token_type_model_attr


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
            print('no types')
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


def load_dataset(lang_path, tokenizer, model, max_length, tagset, dataset_name="test"):
    """Loads conllu file, returns a list of dictionaries (one for each sentence) and a TF dataset"""
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
 
    data = read_conll(glob.glob(lang_path + "/*{}.conllu".format(dataset_name.split("_")[0]))[0])

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
