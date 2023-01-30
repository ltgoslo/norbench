from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np
import datetime
import logging
import torch


def tokenization_func(doc, max_len, tokenizer):
  return tokenizer.encode_plus(
                            doc,  # document to encode.
                            add_special_tokens=True,  # add tokens relative to model
                            max_length=max_len,  # set max length
                            truncation=True,  # truncate longer messages
                            padding='max_length',  # add padding
                            return_attention_mask=True,  # create attn. masks
                            return_token_type_ids=False,
                            return_tensors='pt'  # return pytorch tensors
                       )
  

class DatasetPreparstion(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, dataset, tokenizer, max_len, target_len, source_text, target_text
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = dataset
        self.text_len = max_len
        self.target_len = target_len
        self.target_text = self.data[target_text]
        self.source_text = self.data[source_text]

    
    def __len__(self):
        """returns the length of dataframe"""

        return len(self.source_text)


    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        source = tokenization_func(source_text, self.text_len, self.tokenizer)
        target = tokenization_func(target_text, self.target_len, self.tokenizer)

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


def tokenize_dataset(df, tokenizer, max_len):
    input_ids = []
    attention_masks = []
    max_len = max_len
    for doc in df:
        encoded_dict = tokenizer.encode_plus(
                            doc,  # document to encode.
                            add_special_tokens=True,  # add tokens relative to model
                            max_length=max_len,  # set max length
                            truncation=True,  # truncate longer messages
                            padding='max_length',  # add padding
                            return_attention_mask=True,  # create attn. masks
                            return_token_type_ids=False,
                            return_tensors='pt'  # return pytorch tensors
                       )

        input_ids.append(encoded_dict['input_ids'])

        # and its attention mask (differentiates padding from non-padding)
        attention_masks.append(encoded_dict['attention_mask'])

    return torch.cat(input_ids, dim=0), torch.cat(attention_masks, dim=0)


def load_dataset(df, tokenizer, max_length, target_len, level):

    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    tqdm.pandas(leave=False)
    # Read data
    df.columns = ["sentiment", "review"]
    df["sentiment"] = pd.to_numeric(df["sentiment"])  # Sometimes label gets read as string

    review_input_ids, review_attention_masks = tokenize_dataset(df["review"].values, tokenizer, max_length)
    if level == 'sentence':
        maping_label = {0:'negative', 1: 'neutral', 2: 'positive'}
    if level == 'document':
        maping_label = {0:'negative', 1: 'fair', 2: 'positive'}
    df['label'] = df['sentiment'].apply(lambda s: maping_label[s])

    return DatasetPreparstion(dataset=df, tokenizer=tokenizer, max_len=max_length, target_len=target_len, source_text='review', target_text='label')


# time function
def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))