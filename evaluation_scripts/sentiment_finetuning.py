from transformers import  AutoModel, AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, T5ForConditionalGeneration
from sklearn.metrics import  classification_report, f1_score, accuracy_score
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from torch import nn
from utils.utils_sentiment_t5 import *
from utils.utils_sentiment import *
import pandas as pd
import numpy as np
import warnings
import pathlib
import torch
import time
import sys 
import os

from transformers import logging
logging.set_verbosity_warning()

warnings.filterwarnings("ignore")

### THE BEGGINING OF IMPLEMENTATION OF ANY MODEL OTHER THAN T5
### =========================================================================================================

class SentimentClassifier(nn.Module):

  def __init__(self, n_classes, custom_wrapper, path_to_model):
    super(SentimentClassifier, self).__init__()

    if not custom_wrapper:
      print('You are using a model from HuggingFace.')
      self.bert = AutoModelForSequenceClassification.from_pretrained(path_to_model, num_labels=n_classes, ignore_mismatched_sizes=True)
    if custom_wrapper:
      print('You are using a custom wrapper, NOT a HuggingFace model.')

      sys.path.append(path_to_model)
      from modeling_norbert import NorbertForSequenceClassification
      self.bert = NorbertForSequenceClassification.from_pretrained(path_to_model, num_labels=n_classes, ignore_mismatched_sizes=True)

  def forward(self, input_ids, attention_mask):

    bert_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask,
      return_dict=True
    )

    logits = bert_output.logits

    return logits

def train_epoch(
  model,
  data_loader,
  loss_fn,
  optimizer,
  device,
  scheduler,
  n_examples
):

  y_true, y_pred = [], []
  model = model.train()
  losses = []
  correct_predictions = 0
  
  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)
    y_true += targets.tolist()
    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    preds_idxs = torch.max(outputs, dim=1).indices
    y_pred += preds_idxs.numpy().tolist()
    loss = loss_fn(outputs, targets)
    correct_predictions += torch.sum(preds_idxs == targets)

    losses.append(loss.item())
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
  f1 = f1_score(y_true, y_pred, average='macro')

  return correct_predictions.double() / n_examples, np.mean(losses), f1


def eval_model(model, data_loader, loss_fn, device, n_examples,level, classification):
  model = model.eval()
  losses = []
  correct_predictions = 0
  y_true, y_pred = [], []
  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)
      y_true += targets.tolist()
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)
      y_pred += preds.tolist()
      loss = loss_fn(outputs, targets)
      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())
  f1 = f1_score(y_true, y_pred, average='macro')
  if level == 'document':
    report = classification_report(labels_to_names_document(y_true), labels_to_names_document(y_pred))
  if level == 'sentence':
    if classification == 2:
        report = classification_report(labels_to_names_sent2(y_true), labels_to_names_sent2(y_pred))
    if classification == 3:
        report = classification_report(labels_to_names_sent3(y_true), labels_to_names_sent3(y_pred))

  return correct_predictions.double() / n_examples, np.mean(losses), f1, report




def training_evaluating_not_t5(level, custom_wrapper, path_to_model, lr, max_length, batch_size, epochs, df_train, df_val, df_test, device, classification):

    lr = int(lr)
    max_length = int(max_length)
    batch_size = int(batch_size)
    epochs = int(epochs)

    tokenizer = AutoTokenizer.from_pretrained(path_to_model)
    train_data_loader = create_data_loader(df_train, tokenizer, max_length, batch_size)
    val_data_loader = create_data_loader(df_val, tokenizer, max_length, batch_size)
    test_data_loader = create_data_loader(df_test, tokenizer, max_length, batch_size)

    class_names = df_train.sentiment.unique()
    model = SentimentClassifier(len(class_names), custom_wrapper, path_to_model)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr))
    total_steps = len(train_data_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(2),
                num_training_steps=total_steps
                )
    
    best_valid_f1 = float('-inf')

    for epoch in range(epochs):
        print(f'---------------------Epoch {epoch + 1}/{epochs}---------------------')
        train_acc, train_loss, train_f1 = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train)
        )
        print()
        print(f'Train loss -- {train_loss} -- accuracy {train_acc} -- f1 {train_f1}')

        val_scores = []
        val_acc, val_loss, val_f1, report = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(df_val),
            level,
            classification
        )
        print()
        print(f'Val loss {val_loss} -- accuracy -- {val_acc} -- f1 {val_f1}')
        print()
        print(report)

        val_scores.append(val_f1)
        if best_valid_f1 < val_f1:
            best_valid_f1 = val_f1
            # save best model
            model_name = path_to_model.split('/')[-1] if path_to_model.split('/')[-1] != '' else path_to_model.split('/')[-2]
            pathlib.Path(f'checkpoints/{level}_{classification}_classes').mkdir(parents=True, exist_ok=True)  
            torch.save(model.state_dict(),f'checkpoints/{level}_{classification}_classes/{model_name}.bin')

    test_acc, test_loss, test_f1, test_report = eval_model(
                                                model,
                                                test_data_loader,
                                                loss_fn,
                                                device,
                                                len(df_test),
                                                level,
                                                classification
                                            )


    print()
    print('-------------TESTINGS-----------------')
    print()
    print(f'Test accuracy {test_acc}, f1 {test_f1}')
    print()
    print(test_report)

    avg_val_f1, avg_test_f1 = max(val_scores), test_f1
    return avg_val_f1, avg_test_f1



### THE BEGGINING OF T5 MODEL IMPLEMENTATION
### =========================================================================================================

def train(model, dataloader, optimizer, scaler, scheduler, tokenizer, device):

    # reset total loss for epoch
    train_total_loss = 0

    # track variables
    total_train_acc = 0
    total_train_f1 = 0
    predictions = []
    actuals = []

    # put model into traning mode
    model.train()
    # for each batch of training data...
    for _, data in enumerate(dataloader, 0):

        b_input_ids = data["source_ids"].to(device)
        b_input_mask = data["source_mask"].to(device)
        b_target_ids = data["target_ids"].to(device)
        b_target_mask = data["target_mask"].to(device)

        # clear previously calculated gradients
        optimizer.zero_grad()

        # runs the forward pass with autocasting.
        with autocast():
            # forward propagation (evaluate model on training batch)
            outputs = model(input_ids=b_input_ids,
                            attention_mask=b_input_mask,
                            labels=b_target_ids,
                            decoder_attention_mask=b_target_mask)

            loss, prediction_scores = outputs[:2]

            train_total_loss += loss.item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        generated_ids = model.generate(
                    input_ids=b_input_ids,
                    attention_mask=b_input_mask,
                    max_length=3
                    )

        preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
        target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in b_target_ids]

        total_train_acc += accuracy_score(target, preds)
        total_train_f1 += f1_score(preds,
                                    target,
                                    average='macro',
                                    labels=np.unique(preds))
        predictions.extend(preds)
        actuals.extend(target)

    avg_train_acc = total_train_acc / len(dataloader)
    avg_train_f1 = total_train_f1 / len(dataloader)
    avg_train_loss = train_total_loss / len(dataloader)

    return avg_train_loss, avg_train_acc, avg_train_f1


def validating(model, dataloader, tokenizer, device):

    # After the completion of each training epoch, measure our performance on
    # our validation set
    pred_df = pd.DataFrame(columns=['true','pred'])

    # put the model in evaluation mode
    model.eval()

    # track variables
    total_valid_loss = 0
    total_valid_acc = 0
    total_valid_f1 = 0
    predictions = []
    actuals = []

    # evaluate data for one epoch
    for _, data in enumerate(dataloader, 0):
        b_input_ids = data["source_ids"].to(device)
        b_input_mask = data["source_mask"].to(device)
        b_target_ids = data["target_ids"].to(device)
        b_target_mask = data["target_mask"].to(device)

        with torch.no_grad():

            # forward propagation (evaluate model on training batch)
            outputs = model(input_ids=b_input_ids,
                            attention_mask=b_input_mask,
                            labels=b_target_ids,
                            decoder_attention_mask=b_target_mask)

            loss, prediction_scores = outputs[:2]

            total_valid_loss += loss.item()

            generated_ids = model.generate(
                    input_ids=b_input_ids,
                    attention_mask=b_input_mask,
                    max_length=3
                    )

            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in b_target_ids]

            total_valid_acc += accuracy_score(target, preds)
            total_valid_f1 += f1_score(preds, target,
                                       average='macro',
                                       labels=np.unique(preds))
            predictions.extend(preds)
            actuals.extend(target)

    pred_df['true'] = actuals
    pred_df['pred'] = predictions
    #pred_df.to_csv('valid_preds.csv')

    # calculate the average loss over all of the batches.
    avg_valid_loss = total_valid_loss / len(dataloader)
    avg_valid_acc = total_valid_acc / len(dataloader)
    avg_valid_f1 = total_valid_f1 / len(dataloader)

    return avg_valid_loss, avg_valid_acc, avg_valid_f1


def testing(model, dataloader, tokenizer,test_dataset, device):

    # put the model in evaluation mode
    model.eval()

    pred_df = pd.DataFrame(columns=['true','pred'])

    # track variables
    total_test_loss = 0
    total_test_acc = 0
    total_test_f1 = 0
    predictions = []
    actuals = []

    # evaluate data for one epoch
    for _, data in enumerate(dataloader, 0):
        b_input_ids = data["source_ids"].to(device)
        b_input_mask = data["source_mask"].to(device)
        b_target_ids = data["target_ids"].to(device)
        b_target_mask = data["target_mask"].to(device)

        with torch.no_grad():

            # forward propagation (evaluate model on training batch)
            outputs = model(input_ids=b_input_ids,
                            attention_mask=b_input_mask,
                            labels=b_target_ids,
                            decoder_attention_mask=b_target_mask)

            loss, prediction_scores = outputs[:2]

            total_test_loss += loss.item()

            generated_ids = model.generate(
                    input_ids=b_input_ids,
                    attention_mask=b_input_mask,
                    max_length=3
                    )

            preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
            target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True) for t in b_target_ids]

            total_test_acc += accuracy_score(target, preds)
            total_test_f1 += f1_score(preds, target,
                                       average='macro',
                                       labels=np.unique(preds))
            predictions.extend(preds)
            actuals.extend(target)

    pred_df['true'] = actuals
    pred_df['pred'] = predictions
    #pred_df.to_csv('test_preds.csv')

    # calculate the average loss over all of the batches.
    avg_test_loss = total_test_loss / len(dataloader)
    avg_test_acc = total_test_acc / len(test_dataset)
    avg_test_f1 = total_test_f1 / len(test_dataset)

    return avg_test_loss, avg_test_acc, avg_test_f1


def training_evaluating_t5(level, path_to_model, lr, max_length, batch_size, epochs, df_train, df_val, df_test, device, classification):

    lr = int(lr)
    max_length = int(max_length)
    batch_size = int(batch_size)
    epochs = int(epochs)

    tokenizer = AutoTokenizer.from_pretrained(path_to_model)

   #########!!!!!!!!!!
    #target_len = classification
    #########!!!!!!!!!!

    train_loader = load_dataset(df_train, tokenizer, max_length=max_length, target_len=3, level=level)
    dev_loader = load_dataset(df_val, tokenizer, max_length=max_length, target_len=3, level=level)
    test_loader = load_dataset(df_test, tokenizer, max_length=max_length, target_len=3, level=level)

    train_dataset = DataLoader(train_loader, batch_size=int(batch_size), drop_last=True, shuffle=True)
    val_dataset = DataLoader(dev_loader, batch_size=int(batch_size), shuffle=False)
    test_dataset = DataLoader(test_loader, batch_size=int(batch_size), shuffle=False)

    model = T5ForConditionalGeneration.from_pretrained(path_to_model).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                    lr = float(lr)
                    )
    # lr scheduler
    total_steps = len(train_dataset) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps= int(2),
                                                num_training_steps=total_steps)
    # create gradient scaler for mixed precision
    scaler = GradScaler()

    valid_losses = []
    best_valid_loss = float('-inf')

    for epoch in range(epochs):
        print(f'---------------------Epoch {epoch + 1}/{epochs}---------------------')
        
        # train
        avg_train_loss, avg_train_acc, avg_train_f1 = train(model,
                    train_dataset,
                    optimizer,
                    scaler,
                    scheduler,
                    tokenizer,
                    device)

        print()
        print(f'Train loss -- {avg_train_loss} -- accuracy {avg_train_acc} -- f1 {avg_train_f1}')
        print()

        # validate
        avg_valid_loss, avg_valid_acc, avg_valid_f1 = validating(model, val_dataset, tokenizer, device)
        print(f'Validation loss -- {avg_valid_loss} -- accuracy {avg_valid_acc} -- f1 {avg_valid_f1}')
        print()
        valid_losses.append(avg_valid_loss)

        # check validation loss
        if valid_losses[epoch] > best_valid_loss:
            best_valid_loss = valid_losses[epoch]
            # save best model for use later
            model_name = path_to_model.split('/')[-1] if path_to_model.split('/')[-1] != '' else path_to_model.split('/')[-2]
            pathlib.Path(f'checkpoints/{level}_{classification}_classes').mkdir(parents=True, exist_ok=True)  
            torch.save(model.state_dict(),f'checkpoints/{level}_{classification}_classes/{model_name}.pt')

    # TEST 
    print()
    print('-------------TESTINGS-----------------')
    print()
    avg_test_loss, avg_test_acc, avg_test_f1 = testing(model, test_dataset, tokenizer, test_dataset, device)
    print(f'Test loss -- {avg_test_loss} -- accuracy {avg_test_acc} -- f1 {avg_test_f1}')

    return  max(valid_losses), avg_test_f1


### General training and evaluating function
### =========================================================================================================


# DONT FORGET TO CHANGE ARGUMENTS AND DELETE DF REDUCTION!!!!
def training_evaluating(task_specific_info:str, path_to_model:str, custom_wrapper:False, lr=1e-05, max_length=64, batch_size=4, epochs=1):
    
    level, classification = task_specific_info.split('_')
    classification = int(classification)

    if level == 'sentence':
        print('You are finetuning sentence-level SA!')

        if classification == 2:
            if os.path.exists('data/sentiment/sentence/2class'):
                df_train, df_val, df_test = find_csv('data/sentiment/sentence/2class')
            else:
                df_train, df_val, df_test = load_data('sentence', classification)
            df_train['sentiment'] = df_train.sentiment.replace(2,1)
            df_val['sentiment'] = df_val.sentiment.replace(2,1)
            df_test['sentiment'] = df_test.sentiment.replace(2,1)
            
        if classification == 3:
            if os.path.exists('data/sentiment/sentence/3class'):
                df_train, df_val, df_test = find_csv('data/sentiment/sentence/3class')
            else:
                df_train, df_val, df_test = load_data('sentence', classification)

    if level == 'document':
        print('You are finetuning document-level SA!')

        if os.path.exists('data/sentiment/document'):
            df_train, df_val, df_test = find_csv('data/sentiment/document')
        else:
            df_train, df_val, df_test = load_data('document', classification)
        
        df_train, df_val, df_test = labels_6_to_3(df_train), labels_6_to_3(df_val), labels_6_to_3(df_test)
    
    if level != 'sentence' and level != 'document': 
        print('Please specify level of sentiment analysis and number of classes! Examples: "sentence_2", "sentence_3" or "document_3"')

    print(f'Train samples: {len(df_train)}')
    print(f'Validation samples: {len(df_val)}')
    print(f'Test samples: {len(df_test)}')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    automodel = AutoModel.from_pretrained(path_to_model)

    if 't5' not in automodel.config.architectures[0].lower():
        avg_val_f1, avg_test_f1 = training_evaluating_not_t5(level, custom_wrapper, path_to_model, lr, max_length, batch_size, epochs, df_train, df_val, df_test, device, classification)
    else:
        avg_val_f1, avg_test_f1 = training_evaluating_t5(level, path_to_model, lr, max_length, batch_size, epochs, df_train, df_val, df_test, device, classification)

    return avg_val_f1, avg_test_f1




#avg_val_f1, avg_test_f1 = training_evaluating(task_specific_info='sentence_2', custom_wrapper=False, path_to_model='t5-small')


