
#!/bin/env python3

import tensorflow as tf
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers.data.processors.utils import InputFeatures


class Example:
    def __init__(self, text, category_index):
        self.text = text
        self.category_index = category_index


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


def convert_examples_to_tf_dataset(
        examples,
        tokenizer,
        model,
        max_length=64,
        token_type_ids_input=True
):
    """
    Loads data into a tf.data.Dataset for finetuning a given model.

    Args:
        examples: List of tuples representing the examples to be fed
        tokenizer: Instance of a tokenizer that will tokenize the examples
        model: Instance of a transformers model that is needed to check the ability to use token_type_ids
        max_length: Maximum string length,
        token_type_ids_input: (bool), parameter to include or exclude token_type_ids

    Returns:
        a ``tf.data.Dataset`` containing the condensed features of the provided sentences
    """
    features = []  # -> will hold InputFeatures to be converted later

    token_type_attr = token_type_model_attr(model, max_length)

    for e in examples:

        if token_type_ids_input == True and token_type_attr == True:
            # Documentation is really strong for this method, so please take a look at it
            input_dict = tokenizer.encode_plus(
                e.text,
                add_special_tokens=True,
                max_length=max_length,  # truncates if len(s) > max_length
                return_token_type_ids=True,
                return_attention_mask=True,
                padding="max_length",  # pads to the right by default
                truncation=True
            )

            # input ids = token indices in the tokenizer's internal dict
            # token_type_ids = binary mask identifying different sequences in the model
            # attention_mask = binary mask indicating the positions of padded tokens
            # so the model does not attend to them

            input_ids, token_type_ids, attention_mask = (input_dict["input_ids"],
                                                        input_dict["token_type_ids"],
                                                        input_dict['attention_mask'])

            features.append(
                InputFeatures(
                    input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                    label=e.category_index
                )
            )
        else:
            input_dict = tokenizer.encode_plus(
                e.text,
                add_special_tokens=True,
                max_length=max_length,  # truncates if len(s) > max_length
                return_attention_mask=True,
                pad_to_max_length=True,  # pads to the right by default
                truncation=True
            )

            input_ids, attention_mask = (input_dict["input_ids"], input_dict['attention_mask'])

            features.append(
                InputFeatures(
                    input_ids=input_ids, attention_mask=attention_mask, label=e.category_index
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
                tf.TensorShape([]),
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
                tf.TensorShape([]),
            ),
        )

def load_dataset(data_path, tokenizer, model, max_length, balanced=False,
                 dataset_name="test", limit=None):
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    tqdm.pandas(leave=False)
    # Read data
    df = pd.read_csv(data_path + "{}.csv".format(dataset_name.split("_")[0]), header=None)
    df.columns = ["sentiment", "review"]
    df["sentiment"] = pd.to_numeric(df["sentiment"])  # Sometimes label gets read as string

    # Remove excessively long examples
    lengths = df["review"].progress_apply(lambda x: len(tokenizer.encode(x)))
    df = df[lengths <= max_length].reset_index(drop=True)  # Remove long examples

    # Balance classes
    if dataset_name == "train" and balanced:
        positive_examples = df["sentiment"].sum()
        if not limit:
            # Find which class is the minority and set its size as limit
            n = min(positive_examples, df.shape[0] - positive_examples)
        else:
            n = limit
        ones_idx = np.random.choice(np.where(df["sentiment"])[0], size=n)
        zeros_idx = np.random.choice(np.where(df["sentiment"] == 0)[0], size=n)
        df = df.loc[list(ones_idx) + list(zeros_idx)].reset_index(drop=True)
    elif not balanced and limit:
        raise Exception("Must set 'balanced' to True to choose a manual limit.")

    # Convert to TF dataset
    token_type_ids_input = False if ('roberta' in tokenizer.name_or_path or 'distilbert' in tokenizer.name_or_path) else True


    try:

        dataset = convert_examples_to_tf_dataset(
            [(Example(text=text, category_index=label)) for label, text in df.values], tokenizer, model,
                max_length=max_length, token_type_ids_input=token_type_ids_input)

    except:
        dataset = convert_examples_to_tf_dataset(
            [(Example(text=text, category_index=label)) for label, text in df.values], tokenizer, model,
                max_length=max_length, token_type_ids_input=False)


    return df, dataset
