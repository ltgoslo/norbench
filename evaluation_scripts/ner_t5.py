#!/bin/env python3

from transformers import XLMRobertaTokenizer, XLMRobertaForTokenClassification, XLMRobertaTokenizerFast, DataCollatorForTokenClassification, Trainer, TrainingArguments
import argparse
import pandas as pd
import os
import sys
import textwrap
from torch.utils.data import Dataset, DataLoader
import torch
import glob
import time
import datetime
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
from torch.cuda.amp import autocast, GradScaler
from utils.model_utils import create_model, download_datasets
from utils.ner_utils import generate_label, find_sub_list
import random as python_random
import data_preparation.data_preparation_ner as data_preparation_ner
import evaluate_ner
from datasets import load_metric
from transformers import (
    AdamW,
    T5Model,
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5TokenizerFast,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)

metric = load_metric("seqeval")
import warnings
from conllu import parse
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "true"
tagset = data_preparation_ner.tagset
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# For reproducibility:
seed = 42
np.random.seed(seed)
python_random.seed(seed)

def getting_data(tokenizer, path, batch_size=8, max_length=512, eval_batch_size=8, full_pipeline=True):
    if full_pipeline == True:
        train_dataset_data, val_dataset_data, test_dataset_data = data_preparation_ner.collecting_data_t5(tokenizer, path, max_length)
        train_dataset = DataLoader(train_dataset_data, batch_size=batch_size, drop_last=True, shuffle=True)
        val_dataset = DataLoader(val_dataset_data, batch_size=eval_batch_size)
        test_dataset = DataLoader(test_dataset_data, batch_size=batch_size)
        return train_dataset, val_dataset, test_dataset
    else:
        test_dataset_data = data_preparation_ner.collecting_data_t5(tokenizer, path, max_length, full_pipeline=False)
        test_dataset = DataLoader(test_dataset_data, batch_size=batch_size)
        return test_dataset


def initialize_opt_shed(model, train_dataset, epochs, learning_rate=3e-5):
    optimizer = AdamW(model.parameters(),
                  lr = learning_rate
                  )

    total_steps = len(train_dataset) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    scaler = GradScaler()
    return optimizer, scheduler, scaler


def train(model, tokenizer, dataloader, optimizer, scaler, scheduler, training_stats, epoch, epochs):

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch + 1, epochs))
    print('Training...')

    train_total_loss = 0

    model.train()
    try:
        progress_bar = tqdm(total=len(dataloader.dataset), desc='Epoch {}'.format(epoch + 1))
        for _, data in enumerate(dataloader, 0):

            b_input_ids = data["source_ids"].to(device)
            b_input_mask = data["source_mask"].to(device)
            b_target_mask = data["target_mask"].to(device)
            lm_labels = data["target_ids"].to(device)

            # lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100

            optimizer.zero_grad()

            # print(b_input_ids.shape, b_input_mask.shape, b_target_mask.shape, lm_labels.shape)
            with autocast():
                outputs = model(input_ids=b_input_ids,
                                attention_mask=b_input_mask,
                                labels=lm_labels,
                                decoder_attention_mask=b_target_mask)

                loss, prediction_scores = outputs[:2]

                train_total_loss += loss.item()

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            scheduler.step()
            
            progress_bar.set_postfix(current_loss=loss.item())
            progress_bar.update(len(b_input_ids))

        progress_bar.close()

    except KeyboardInterrupt:

        progress_bar.close()

    avg_train_loss = train_total_loss / len(dataloader)

    print('\n\nMean Loss after epoch #{0} - {1}'.format(str(epoch + 1), avg_train_loss))

    return model


def validating(model, tokenizer, dataloader, valid_stats, epoch):


    print("")
    print("Validation...")

    model.eval()

    total_valid_loss = 0
    try:
        progress_bar = tqdm(total=len(dataloader.dataset), desc='Epoch {}'.format(epoch + 1))
        for _, data in enumerate(dataloader, 0):

            b_input_ids = data["source_ids"].to(device)
            b_input_mask = data["source_mask"].to(device)
            lm_labels = data["target_ids"].to(device)
            # lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
            b_target_mask = data["target_mask"].to(device)

            with torch.no_grad():

                outputs = model(input_ids=b_input_ids,
                                attention_mask=b_input_mask,
                                labels=lm_labels,
                                decoder_attention_mask=b_target_mask)

                loss, prediction_scores = outputs[:2]

                total_valid_loss += loss.item()
            
            progress_bar.set_postfix(current_loss=loss.item())
            progress_bar.update(len(b_input_ids))

        progress_bar.close()

    except KeyboardInterrupt:

        progress_bar.close()

    global avg_val_loss
    avg_val_loss = total_valid_loss / len(dataloader)

    print('\n\nMean Loss after epoch #{0} - {1} on validation'.format(str(epoch + 1), avg_val_loss))
    valid_stats.append(
        {
            'Val Loss': avg_val_loss,
        }
    )

    return valid_stats



def train_evaluation(data_path, sub_info, model_name, run_name, task, epochs, use_seqeval_evaluation, max_length=512, batch_size=8, eval_batch_size=8, learning_rate=3e-5, custom_wrapper=False, tagset=tagset):

    checkpoints_path = f"checkpoints/{task}/{sub_info}/{run_name}/"
    if not os.path.isdir(checkpoints_path):
          os.makedirs(checkpoints_path)

    if data_path == True:
        data_path = download_datasets(task, sub_info)

    tokenizer = AutoTokenizer.from_pretrained(model_name) 
    if not custom_wrapper:
        # tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    else:
        sys.path.append(model_name)
        print('You are using a custom wrapper, NOT a HuggingFace model.')
        from modeling_nort5 import NorT5ForConditionalGeneration
        # tokenizer = AutoTokenizer.from_pretrained(model_name) 
        # tokenizer = T5TokenizerFast.from_pretrained(model_name)
        model = NorT5ForConditionalGeneration.from_pretrained(model_name).to(device)


    train_dataset, val_dataset, test_dataset  =  getting_data(tokenizer, data_path, batch_size=batch_size, max_length=max_length, eval_batch_size=eval_batch_size)
    optimizer, scheduler, scaler = initialize_opt_shed(model, train_dataset, epochs, learning_rate)
    training_stats = []
    valid_stats = []
    best_valid_loss = float('inf')


    for epoch in range(epochs):

        train(model, tokenizer, train_dataset, optimizer, scaler, scheduler, training_stats, epoch, epochs)

        validating(model, tokenizer, val_dataset, valid_stats, epoch)

        if valid_stats[epoch]['Val Loss'] < best_valid_loss:
            best_valid_loss = valid_stats[epoch]['Val Loss']
            print(f"\nBest validation loss: {best_valid_loss}")
            name_to_save = f'{run_name}_{task}'
            if os.path.isfile(checkpoints_path+name_to_save+'.pth'):
              os.remove(checkpoints_path+name_to_save+'.pth')
              torch.save(model.state_dict(), checkpoints_path+name_to_save+'.pth')
            else:
              torch.save(model.state_dict(), checkpoints_path+name_to_save+'.pth')
    
    return model, tokenizer



def test_generate(model, tokenizer, test_dataset):

    model.eval()
    outputs = []
    targets = []
    all_text = []
    true_labels = []
    pred_labels = []
    for batch in tqdm(test_dataset):
        input_ids = batch['source_ids'].to(device)
        attention_mask = batch['source_mask'].to(device)
        outs = model.generate(input_ids=input_ids,
                                    attention_mask=attention_mask)
        dec = [tokenizer.decode(ids, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False).strip() for ids in outs]
        target = [tokenizer.decode(ids, skip_special_tokens=True,  clean_up_tokenization_spaces=False).strip()
                    for ids in batch["target_ids"]]
        texts = [tokenizer.decode(ids, skip_special_tokens=True,  clean_up_tokenization_spaces=False).strip()
                    for ids in batch["source_ids"]]
        true_label = [generate_label(texts[i].strip(), target[i].strip()) if target[i].strip() != 'none' else [
            "O"]*len(texts[i].strip().split()) for i in range(len(texts))]
        pred_label = [generate_label(texts[i].strip(), dec[i].strip()) if dec[i].strip() != 'none' else [
            "O"]*len(texts[i].strip().split()) for i in range(len(texts))]

        outputs.extend(dec)
        targets.extend(target)
        true_labels.extend(true_label)
        pred_labels.extend(pred_label)
        all_text.extend(texts)
    
    return outputs, targets, all_text, true_labels, pred_labels


def test(data_path, name_sub_info, model_identifier, tokenizer, current_task, run_name, batch_size=8, max_length=512, tagset=tagset, custom_wrapper=False):
    
    if data_path == True:
        data_path = download_datasets(current_task, name_sub_info)
    
    checkpoints_path = f"checkpoints/{current_task}/{name_sub_info}/{run_name}/"
    name_to_save = f'{run_name}_{current_task}'+ '.pth'

    if not custom_wrapper:
        model = T5ForConditionalGeneration.from_pretrained(model_identifier).to(device)
    else:
        sys.path.append(model_identifier)
        from modeling_nort5 import NorT5ForConditionalGeneration
        model = NorT5ForConditionalGeneration.from_pretrained(model_identifier).to(device)

    model.load_state_dict(torch.load(checkpoints_path+name_to_save))
    test_dataset = getting_data(tokenizer, data_path, max_length=max_length, batch_size=batch_size, full_pipeline=False)
    outputs, targets, all_text, true_labels, pred_labels = test_generate(model, tokenizer, test_dataset)
    
    
    print('PREPARING TO SAVE PREDICTIONS')

    path_to_test = glob.glob(data_path + "/*{}.conllu".format('test'.split("_")[0]))[0]
    path_to_predictions = data_path + f"predicted_{run_name}.conllu"

    test_conll = parse(open(path_to_test, "r").read())

    for nr, sentence in enumerate(test_conll):
        for tk_num, token in enumerate(sentence):   
            if len([token['lemma'] for token in sentence]) != len(all_text[nr].split(' ')):
                if sentence[tk_num]['lemma'] == 'å':
                    if all_text[nr].split(' ')[tk_num] != 'å':
                      pred_labels[nr].insert(tk_num, 'O')
            try:
                token["misc"]["name"] = pred_labels[nr][tk_num]
            except:
                token["misc"]["name"] = 'O'

    with open(path_to_predictions, "w") as f:
            for sentence in test_conll:
                f.write(sentence.serialize())
    print('Saving file')

    print('Scores:')
    test_results =  evaluate_ner.evaluation(path_to_predictions, path_to_test)
    return test_results
