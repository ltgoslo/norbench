from transformers import  AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, T5ForConditionalGeneration
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
import datetime
import pathlib
import logging
import random
import torch
import sys
import os
import tqdm

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def seed_everything(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


### THE BEGGINING OF IMPLEMENTATION OF ANY MODEL OTHER THAN T5
### =========================================================================================================

class SentimentClassifier(nn.Module):

  def __init__(self, n_classes, custom_wrapper, model_identifier):
    super(SentimentClassifier, self).__init__()

    if not custom_wrapper:
      logger.info('You are using a model from HuggingFace.')
      self.bert = AutoModelForSequenceClassification.from_pretrained(model_identifier,
                                                                     num_labels=n_classes,
                                                                     ignore_mismatched_sizes=True)
    if custom_wrapper:
      logger.info('You are using a custom wrapper, NOT a HuggingFace model.')

      sys.path.append(model_identifier)
      from modeling_norbert import NorbertForSequenceClassification
      self.bert = NorbertForSequenceClassification.from_pretrained(model_identifier, num_labels=n_classes,
                                                                   ignore_mismatched_sizes=True)

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
  
  for d in tqdm.tqdm(data_loader):
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)
    y_true += targets.tolist()
    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    preds_idxs = torch.max(outputs, dim=1).indices
    y_pred += preds_idxs.cpu().numpy().tolist()
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


def eval_model(model, data_loader, loss_fn, device, n_examples, level):
  pred_df = pd.DataFrame(columns=['true','pred'])
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
    report = classification_report(labels_to_names(y_true), labels_to_names(y_pred))
    pred_df['true'] = labels_to_names(y_true)
    pred_df['pred'] = labels_to_names(y_pred)
  if level == 'sentence':
    report = classification_report(labels_to_names_sentence(y_true), labels_to_names_sentence(y_pred))
    pred_df['true'] = labels_to_names_sentence(y_true)
    pred_df['pred'] = labels_to_names_sentence(y_pred)
  if level == 'other':
    report = classification_report(y_true, y_pred)
    pred_df['true'] = y_true
    pred_df['pred'] = y_pred
  return correct_predictions.double() / n_examples, np.mean(losses), f1, report, pred_df


def training_evaluating_not_t5(df_train, df_val, df_test, level, model_identifier, run_name, epochs, max_length, batch_size, learning_rate, custom_wrapper, seed):

    logger.info(f'Train samples: {len(df_train)}')
    logger.info(f'Validation samples: {len(df_val)}')
    logger.info(f'Test samples: {len(df_test)}')


    tokenizer = AutoTokenizer.from_pretrained(model_identifier)
    train_data_loader = create_data_loader(df_train, tokenizer, max_length, batch_size)
    val_data_loader = create_data_loader(df_val, tokenizer, max_length, batch_size)
    test_data_loader = create_data_loader(df_test, tokenizer, max_length, batch_size)

    class_names = df_train.sentiment.unique()
    model = SentimentClassifier(len(class_names), custom_wrapper, model_identifier)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(learning_rate))
    total_steps = len(train_data_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=2,
                num_training_steps=total_steps
                )
    
    best_valid_f1 = float('inf')

    for epoch in range(epochs):
        logger.info(f'---------------------Epoch {epoch + 1}/{epochs}---------------------')
        train_acc, train_loss, train_f1 = train_epoch(
            model,
            train_data_loader,
            loss_fn,
            optimizer,
            device,
            scheduler,
            len(df_train)
        )
        logger.info(f'Train loss -- {train_loss} -- accuracy {train_acc} -- f1 {train_f1}')

        val_scores = []
        val_acc, val_loss, val_f1, report, _ = eval_model(
            model,
            val_data_loader,
            loss_fn,
            device,
            len(df_val),
            level
        )
        logger.info(f'Val loss {val_loss} -- accuracy -- {val_acc} -- f1 {val_f1}')
        logger.info(report)

        val_scores.append(val_f1)
        if best_valid_f1 <= val_f1:
            best_valid_f1 = val_f1
            # save best model
            pathlib.Path('./saved_models').mkdir(parents=True, exist_ok=True)  
            torch.save(model.state_dict(),f'saved_models/{run_name}.bin')
        if best_valid_f1 > val_f1:
            break

    logger.info('-------------TESTINGS-----------------')
    model = SentimentClassifier(len(class_names), custom_wrapper, model_identifier)
    model.load_state_dict(torch.load(f'saved_models/{run_name}.bin'))
    test_acc, test_loss, test_f1, test_report, pred_df = eval_model(
                                                model,
                                                test_data_loader,
                                                loss_fn,
                                                device,
                                                len(df_test),
                                                level
                                            )

    pathlib.Path('./test_predictions').mkdir(parents=True, exist_ok=True) 
    pred_df = pred_df.assign(text=df_test.review)
    pred_df.to_csv(f'./test_predictions/{run_name}.csv')
    logger.info(f'Test accuracy {test_acc}, f1 {test_f1}')
    logger.info(test_report)
    print(f"{datetime.datetime.now()}\t{seed}\t{test_acc:.4f}\t{test_f1:.4f}")
    return best_valid_f1, test_f1


### THE BEGGINING OF T5 MODEL IMPLEMENTATION
### =========================================================================================================

def train(model, dataloader, optimizer, scaler, scheduler, tokenizer):

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
    for _, data in tqdm.tqdm(enumerate(dataloader, 0)):

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


def validating(model, dataloader, tokenizer):

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

    # calculate the average loss over all of the batches.
    avg_valid_loss = total_valid_loss / len(dataloader)
    avg_valid_acc = total_valid_acc / len(dataloader)
    avg_valid_f1 = total_valid_f1 / len(dataloader)

    return avg_valid_loss, avg_valid_acc, avg_valid_f1

def testing(model, dataloader, tokenizer,test_dataset):

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

    # calculate the average loss over all of the batches.
    avg_test_loss = total_test_loss / len(dataloader)
    avg_test_acc = total_test_acc / len(test_dataset)
    avg_test_f1 = total_test_f1 / len(test_dataset)

    return avg_test_loss, avg_test_acc, avg_test_f1, pred_df

def training_evaluating_t5(df_train, df_val, df_test, level, model_identifier, run_name, epochs, max_length, batch_size, learning_rate, custom_wrapper, seed):

    logger.info(f'Train samples: {len(df_train)}')
    logger.info(f'Validation samples: {len(df_val)}')
    logger.info(f'Test samples: {len(df_test)}')

    tokenizer = AutoTokenizer.from_pretrained(model_identifier)

    train_loader = load_dataset(df_train, tokenizer, max_length=max_length, target_len=3, level=level)
    dev_loader = load_dataset(df_val, tokenizer, max_length=max_length, target_len=3, level=level)
    test_loader = load_dataset(df_test, tokenizer, max_length=max_length, target_len=3, level=level)

    train_dataset = DataLoader(train_loader, batch_size=int(batch_size), drop_last=True, shuffle=True)
    val_dataset = DataLoader(dev_loader, batch_size=int(batch_size), shuffle=False)
    test_dataset = DataLoader(test_loader, batch_size=int(batch_size), shuffle=False)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    if custom_wrapper==False:
      model = T5ForConditionalGeneration.from_pretrained(model_identifier).to(device)
    if custom_wrapper==True:
      sys.path.append(model_identifier)
      from modeling_nort5 import NorT5ForConditionalGeneration
      model = NorT5ForConditionalGeneration.from_pretrained(model_identifier).to(device)


    optimizer = torch.optim.AdamW(model.parameters(),
                    lr = float(learning_rate)
                    )
    # epochs
    epochs = int(epochs)
    # lr scheduler
    total_steps = len(train_dataset) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps= 2,
                                                num_training_steps=total_steps)
    # create gradient scaler for mixed precision
    scaler = GradScaler()

    valid_losses = []
    best_valid_loss = float('inf')

    for epoch in range(epochs):
        logger.info(f'---------------------Epoch {epoch + 1}/{epochs}---------------------')
        
        # train
        avg_train_loss, avg_train_acc, avg_train_f1 = train(model,
                    train_dataset,
                    optimizer,
                    scaler,
                    scheduler,
                    tokenizer)

        logger.info(f'Train loss -- {avg_train_loss} -- accuracy {avg_train_acc} -- f1 {avg_train_f1}')

        # validate
        avg_valid_loss, avg_valid_acc, avg_valid_f1 = validating(model, val_dataset, tokenizer)
        logger.info(f'Validation loss -- {avg_valid_loss} -- accuracy {avg_valid_acc} -- f1 {avg_valid_f1}')
        valid_losses.append(avg_valid_loss)
        # check validation loss
        # EARLY STOPPING
        print('AVG: ',avg_valid_loss)
        print('BEST: ', best_valid_loss)
        if avg_valid_loss <= best_valid_loss:
            best_valid_loss = avg_valid_loss
            # save best model for use later
            pathlib.Path('./saved_models').mkdir(parents=True, exist_ok=True)  
            torch.save(model.state_dict(),f'saved_models/{run_name}.pt')
        else: 
            break
        

    # TEST 
    logger.info('-------------TESTINGS-----------------')

    if custom_wrapper==False:
      model = T5ForConditionalGeneration.from_pretrained(model_identifier).to(device)
    if custom_wrapper==True:
      sys.path.append(model_identifier)
      from modeling_nort5 import NorT5ForConditionalGeneration
      model = NorT5ForConditionalGeneration.from_pretrained(model_identifier).to(device)

    model.load_state_dict(torch.load(f'./saved_models/{run_name}.pt'))
    avg_test_loss, avg_test_acc, avg_test_f1, pred_df = testing(model, test_dataset, tokenizer, test_dataset)
    
    pathlib.Path('./test_predictions').mkdir(parents=True, exist_ok=True) 
    pred_df = pred_df.assign(text=df_test.review)
    pred_df.to_csv(f'./test_predictions/{run_name}.csv')
    logger.info(f'Test loss -- {avg_test_loss} -- accuracy {avg_test_acc} -- f1 {avg_test_f1}')
    return avg_valid_f1, avg_test_f1


### General training and evaluating function
### =========================================================================================================

def training_evaluating(
    check_for_t5 = False,
    name_sub_info = 'sentence',
    data_path = True, 
    model_identifier = 'ltg/norbert2',
    run_name = 'norbench_model',
    epochs = 10,
    max_length = 256,
    batch_size = 8,
    learning_rate = 2e-05,
    custom_wrapper = False,
    seed = 42
):
    seed_everything(seed)
    # load train, test and dev datasets 
    level = name_sub_info
    if level == 'sentence' or level == 'document':
        logger.info(f'You are finetuning {level}-level SA!')
        if os.path.exists(f'data/{level}'):
            df_train, df_val, df_test = find_csv(f'data/{level}') 
        else:
            df_train, df_val, df_test = load_data(level)
    if level == 'document':
        df_train, df_val, df_test = labels_6_to_3(df_train), labels_6_to_3(df_val), labels_6_to_3(df_test)
    if data_path != True and level == 'other':
        df_train, df_val, df_test = find_csv(data_path)

    if level != 'document' and level != 'sentence' and level != 'other':
        logger.info("Please specify --task_specific_info argument: 'sentence' if you want to use corpora with "
                    "sentence-level sentiment analysys or 'document' for document-level SA. 'other' "
                    "if you want to use your own corpora")
        # Exit the script with a non-zero exit code to indicate an error
        sys.exit(1)

    #  training and evaluation for masked language models
    if check_for_t5 == False:
        dev_score, test_score = training_evaluating_not_t5(df_train, df_val, df_test, level, model_identifier, run_name, epochs, max_length, batch_size, learning_rate, custom_wrapper, seed)

     #  training and evaluation T5 models
    if check_for_t5 == True:
        dev_score, test_score = training_evaluating_t5(df_train, df_val, df_test, level, model_identifier, run_name, epochs, max_length, batch_size, learning_rate, custom_wrapper, seed)


    return dev_score, test_score
