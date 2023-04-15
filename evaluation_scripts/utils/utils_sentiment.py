from torch.utils.data import Dataset, DataLoader
from git import Repo
import pandas as pd
import numpy as np
import pathlib
import torch
import json
import os

def labels_6_to_3(df):

    df.sentiment = df.sentiment.replace(1,0)
    df.sentiment = df.sentiment.replace(2,0)
    df.sentiment = df.sentiment.replace(3,1)
    df.sentiment = df.sentiment.replace(4,2)
    df.sentiment = df.sentiment.replace(5,2)

    return df

def labels_to_names(labels):

    mapping = {
        0:'Negative',
        1:'Fair',
        2:'Positive'
    }
    new_labels = [mapping[label] for label in labels]

    return new_labels


def labels_to_names_sentence(labels):

    mapping = {
        0:'Negative',
        1:'Neutral',
        2:'Positive'
    }
    new_labels = [mapping[label] for label in labels]

    return new_labels

def get_text(path: str) -> str:

    with open(path, 'r' ,encoding='utf-8') as f:
        text = f.read()

    return text

def create_dataframe_sentence(path):

    with open(path, 'r') as j:
        js = json.loads(j.read())

    texts, labels = [], []

    for sample in js:
        texts.append(sample['text'])
        labels.append(sample['label'])

    encoded_labels = []
    for i in labels:
      if i == 'Negative':
        encoded_labels.append(0)
      if i == 'Neutral':
        encoded_labels.append(1)
      if i == 'Positive':
        encoded_labels.append(2)

    df = pd.DataFrame(columns=["sentiment", "review"])
    df['sentiment'] = encoded_labels
    df['review'] = texts
    return df

def find_csv(path):
    needed_csv = ['train.csv', 'test.csv', 'dev.csv']
    for csv in needed_csv:
        if not os.path.isfile(os.path.join(path,csv)):
            print('Could not find train.csv, test.csv or dev.csv in your diectory!')
            return None

    df_train = pd.read_csv(os.path.join(path,'train.csv'))
    df_val = pd.read_csv(os.path.join(path,'dev.csv'))
    df_test = pd.read_csv(os.path.join(path,'test.csv'))

    return df_train, df_val, df_test

# level: sentence or document
def load_data(level):

    if level == 'document':
        norec_document_link = 'https://github.com/ltgoslo/norec'
        if not os.path.exists('data/norec/'):
            Repo.clone_from(norec_document_link, 'data/norec')
        path = 'data/norec/data'


        train = pd.DataFrame(columns=['review', 'sentiment'])
        test = pd.DataFrame(columns=['review', 'sentiment'])
        dev = pd.DataFrame(columns=['review', 'sentiment'])

        with open(os.path.join(path, "metadata.json"), 'r') as js:
            metadata = json.load(js)

        for k,v in metadata.items():
            split = v['split']
            review = get_text(os.path.join(path, f"{split}/{k}.txt"))
            sentiment = int(v['rating'])-1
            if split == 'train':
                train = train.append({'review':review, 'sentiment':sentiment },ignore_index=True)
            if split == 'test':
                test = test.append({'review':review, 'sentiment':sentiment }, ignore_index=True)
            if split == 'dev':
                dev = dev.append({'review':review, 'sentiment':sentiment },ignore_index=True)

        pathlib.Path('./data/document').mkdir(parents=True, exist_ok=True)
        train[['sentiment', 'review']].to_csv('data/document/train.csv', index=False)
        dev[['sentiment', 'review']].to_csv('data/document/dev.csv', index=False)
        test[['sentiment', 'review']].to_csv('data/document/test.csv', index=False)

    if level == 'sentence':
        norec_sentence_link = 'https://github.com/ltgoslo/norec_sentence'
        if not os.path.exists('data/norec_sentence/3class/'):
            Repo.clone_from(norec_sentence_link, 'data/norec_sentence')
        path = 'data/norec_sentence/3class/'

        train = create_dataframe_sentence(os.path.join(path, "train.json"))
        dev = create_dataframe_sentence(os.path.join(path, "dev.json"))
        test = create_dataframe_sentence(os.path.join(path, "test.json"))

        pathlib.Path('./data/sentence').mkdir(parents=True, exist_ok=True)
        train.to_csv('data/sentence/train.csv', index=False)
        dev.to_csv('data/sentence/dev.csv', index=False)
        test.to_csv('data/sentence/test.csv', index=False)

    return train, dev, test

class Dataset(Dataset):
  def __init__(self, texts, targets, tokenizer, max_len):
    self.texts = texts
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, item):
    text = str(self.texts[item])
    target = self.targets[item]
    encoding = self.tokenizer(
      text,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      padding="longest",
      return_attention_mask=True,
      truncation=True,
      return_tensors='pt',
    )

    return {
      'text': text,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

def create_data_loader(df, tokenizer, max_len, batch_size, shuffle=False):
  ds = Dataset(
    texts=df.review.to_numpy(),
    targets=df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )
  return DataLoader(
    ds,
    batch_size=batch_size,
    shuffle=shuffle
  )

