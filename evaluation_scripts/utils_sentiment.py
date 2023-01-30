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


def labels_to_names_document(labels):

    mapping = {
        0:'Negative',
        1:'Fair',
        2:'Positive'
    }
    new_labels = [mapping[label] for label in labels]

    return new_labels


def labels_to_names_sent2(labels):

    mapping = {
        0:'Negative',
        1:'Positive',
    }
    new_labels = [mapping[label] for label in labels]

    return new_labels

def labels_to_names_sent3(labels):

    mapping = {
        0:'Negative',
        1:'Neutral',
        2:'Positive',
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

# level: sentence or document
def load_data(level:str, classification:int):
    
    if level == 'document':    
        norec_document_link = 'https://github.com/ltgoslo/norec'
        if not os.path.exists('data/sentiment/norec/'):
            Repo.clone_from(norec_document_link, 'data/sentiment/norec/')
        path = 'data/sentiment/norec/data'

        metadata = pd.read_json(os.path.join(path, "metadata.json"), encoding='utf-8')
        data = metadata.T[['id', 'rating']]
        data['txt_names'] = ['0'*(6-len(str(id))) + str(id) + '.txt' for id in data['id']]
        texts = []
        splits = []

        for root, _, files in os.walk(path, topdown=False):
            for name in files:
                path_text = os.path.join(root, name)
                if path_text.endswith('.txt'):
                        texts.append(get_text(path_text))
                        splits.append(path_text.split('/')[-2])

        data['text'] = texts
        data['split'] = splits
        data = data.rename(columns={"rating": "sentiment", 'text':'review'}, errors="raise")
        data['sentiment'] = [x-1 for x in data['sentiment']]

        train = data[data.split == 'train']
        dev = data[data.split == 'dev']
        test = data[data.split == 'test']

        pathlib.Path('data/sentiment/document').mkdir(parents=True, exist_ok=True)
        train[['sentiment', 'review']].to_csv('data/sentiment/document/train.csv', index=False)
        dev[['sentiment', 'review']].to_csv('data/sentiment/document/dev.csv', index=False)
        test[['sentiment', 'review']].to_csv('data/sentiment/document/test.csv', index=False)
    
    if level == 'sentence':
        norec_sentence_link = 'https://github.com/ltgoslo/norec_sentence'
        if not os.path.exists('data/sentiment/norec_sentence/'):
            Repo.clone_from(norec_sentence_link, 'data/sentiment/norec_sentence/')
        
        if classification == 3:
            path = 'data/sentiment/norec_sentence/3class/'
            
            train = create_dataframe_sentence(os.path.join(path, "train.json"))
            dev = create_dataframe_sentence(os.path.join(path, "dev.json"))
            test = create_dataframe_sentence(os.path.join(path, "test.json"))
            
            pathlib.Path('data/sentiment/sentence/3class').mkdir(parents=True, exist_ok=True)
            train.to_csv('data/sentiment/sentence/3class/train.csv', index=False)
            dev.to_csv('data/sentiment/sentence/3class/dev.csv', index=False)
            test.to_csv('data/sentiment/sentence/3class/test.csv', index=False)
        
        if classification == 2:
            path = 'data/sentiment/norec_sentence/binary/'
            
            train = create_dataframe_sentence(os.path.join(path, "train.json"))
            dev = create_dataframe_sentence(os.path.join(path, "dev.json"))
            test = create_dataframe_sentence(os.path.join(path, "test.json"))
            
            pathlib.Path('data/sentiment/sentence/2class').mkdir(parents=True, exist_ok=True)
            train.to_csv('data/sentiment/sentence/2class/train.csv', index=False)
            dev.to_csv('data/sentiment/sentence/2class/dev.csv', index=False)
            test.to_csv('data/sentiment/sentence/2class/test.csv', index=False)
    
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
    encoding = self.tokenizer.encode_plus(
      text,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      padding='max_length',
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

def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = Dataset(
    texts=df.review.to_numpy(),
    targets=df.sentiment.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )
  return DataLoader(
    ds,
    batch_size=batch_size
  )


def find_csv(path):

    df_train = pd.read_csv(os.path.join(path,'train.csv'))
    df_val = pd.read_csv(os.path.join(path,'test.csv'))
    df_test = pd.read_csv(os.path.join(path,'dev.csv'))

    return df_train, df_val, df_test