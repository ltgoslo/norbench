from transformers import XLMRobertaTokenizer, XLMRobertaForTokenClassification, XLMRobertaTokenizerFast, DataCollatorForTokenClassification, Trainer, TrainingArguments
import os
from torch.utils.data import Dataset, DataLoader
import torch
import re
import numpy as np
from tqdm.auto import tqdm
#import tensorflow as tf
from torch.cuda.amp import autocast, GradScaler
from utils.model_utils import create_model, download_datasets
import random as python_random
from utils.pos_utils import entities_tokens
from data_preparation.data_preparation_pos import collecting_data_t5
from transformers import(
    AdamW,
    T5Model,
    T5ForConditionalGeneration,
    T5Tokenizer,
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
import torch
from seqeval.metrics import accuracy_score
import warnings
from conllu import parse
warnings.filterwarnings("ignore")


# the same approach as in https://github.com/israelcamp/T5-For-NER was used


os.environ["TOKENIZERS_PARALLELISM"] = "true"
pos_tags, marked_tags, labels2words, entities2markedtags = entities_tokens()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# For reproducibility:
seed = 42
np.random.seed(seed)
python_random.seed(seed)
#tf.random.set_seed(seed)

pos_tags, marked_tags, labels2words, entities2markedtags = entities_tokens()    
    

def getting_data(tokenizer, path, batch_size=8, max_length=512, eval_batch_size=8, full_pipeline=True):
    if full_pipeline == True:
        train_dataset_data, val_dataset_data, test_dataset_data = collecting_data_t5(tokenizer, path, max_length)
        train_dataset = DataLoader(train_dataset_data, batch_size=batch_size, shuffle=False)
        val_dataset = DataLoader(val_dataset_data, batch_size=eval_batch_size)
        test_dataset = DataLoader(test_dataset_data, batch_size=batch_size)
        return train_dataset, val_dataset, test_dataset
    else:
        test_dataset_data = collecting_data_t5(tokenizer, path, max_length, full_pipeline=False)
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

    progress_bar = tqdm(total=len(dataloader.dataset), desc='Epoch {}'.format(epoch + 1))
    for data in dataloader:
            
        b_input_ids = data["source_ids"].to(device)
        b_input_mask = data["attention_mask"].to(device)
        lm_labels = data["target_ids"].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids=b_input_ids,
                                  attention_mask=b_input_mask,
                                  labels=lm_labels)

        loss, prediction_scores = outputs[:2]

        train_total_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        model.zero_grad()
        scaler.update()

        scheduler.step()
            
        progress_bar.set_postfix(current_loss=loss.item())
        progress_bar.update(len(b_input_ids))

    progress_bar.close()

    avg_train_loss = train_total_loss / len(dataloader)

    print('\n\nMean Loss after epoch #{0} - {1}'.format(str(epoch + 1), avg_train_loss))

    return model


def validating(model, tokenizer, dataloader, valid_stats, epoch):


    print("")
    print("Validation...")

    model.eval()

    total_valid_loss = 0
    for data in dataloader:

      b_input_ids = data["source_ids"].to(device)
      b_input_mask = data["attention_mask"].to(device)
      lm_labels = data["target_ids"].to(device)

      with torch.no_grad():
          outputs = model(input_ids=b_input_ids,
                                attention_mask=b_input_mask,
                                labels=lm_labels)

          loss, prediction_scores = outputs[:2]

      
      total_valid_loss += loss.item()

    global avg_val_loss
    avg_val_loss = total_valid_loss / len(dataloader)

    print('\n\nMean Loss after epoch #{0} - {1} on validation'.format(str(epoch + 1), avg_val_loss))
    valid_stats.append(
        {
            'Val Loss': avg_val_loss,
        }
    )

    return valid_stats


def train_evaluation(data_path, sub_info, model_name, run_name, task, epochs, max_length=512, batch_size=8, eval_batch_size=8, learning_rate=3e-5):

    checkpoints_path = f"checkpoints/{task}/{sub_info}/{run_name}/"
    if not os.path.isdir(checkpoints_path):
          os.makedirs(checkpoints_path)

    if data_path == True:
        data_path = download_datasets(task, sub_info)
    
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    tokenizer.add_tokens(marked_tags)
    tokenizer.add_special_tokens({'eos_token': '<EOS>'})
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    train_dataset, val_dataset, test_dataset  =  getting_data(tokenizer, data_path, batch_size=batch_size, max_length=max_length, eval_batch_size=eval_batch_size)
    optimizer, scheduler, scaler = initialize_opt_shed(model, train_dataset, epochs, learning_rate)
    training_stats = []
    valid_stats = []
    best_valid_loss = float('inf')

    for epoch in range(epochs):
        try:
            
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
        except KeyboardInterrupt:
            break
    return model, tokenizer

    
def generate_labels(true_tokens, pred_tokens):

    tags = labels2words.keys()

    true_tokens =  re.split(r'[\<|\>]', true_tokens.strip())
    pred_tokens =  re.split(r'[\<|\>]', pred_tokens.strip())
    
    upd_true_tokens = [(true_tokens[ind-1], el) for ind, el in enumerate(true_tokens) if el in tags]
    upd_pred_tokens = [(pred_tokens[ind-1], el) for ind, el in enumerate(pred_tokens) if el in tags]

    new_origs_list = [el[1] for el in upd_true_tokens]
    new_prediction_list = ['_']*len(upd_true_tokens)
    upd_ind = None
    last_upd_ind = 0
    
    for ind, tr_lab in enumerate(upd_true_tokens):
      token, cur_tag = tr_lab
        
      if upd_ind == ind and ind + 1 <= len(upd_pred_tokens):
          
          token_pred, cur_tag_pred = upd_pred_tokens[ind]

          if token_pred.strip() == token.strip():
            new_prediction_list[ind] = cur_tag_pred
            upd_ind = ind + 1
            last_upd_ind = ind

      else:
          if upd_ind == None:
            upd_ind = last_upd_ind
          if upd_ind <= len(upd_pred_tokens):
            succes = False
            for new_loop_ind, new_upd_tok in enumerate(upd_pred_tokens[upd_ind:]):
              if new_upd_tok[0].strip() == token.strip():
                new_prediction_list[ind] = new_upd_tok[1].strip()
                last_upd_ind = upd_ind
                upd_ind = upd_ind + new_loop_ind + 1
                succes = True
                break
            if succes == False:
              for new_loop_ind_prev, new_upd_tok_prev in enumerate(upd_pred_tokens[last_upd_ind+1:]):
                  if new_upd_tok_prev[0].strip() == token.strip():
                     new_prediction_list[ind] = new_upd_tok_prev[1].strip()
                     last_upd_ind = last_upd_ind + new_loop_ind_prev + 1
                     upd_ind = last_upd_ind + 1 + new_loop_ind_prev + 1
                     break

    return new_origs_list, new_prediction_list
      

def test(data_path, name_sub_info, model_identifier, tokenizer, current_task, run_name, batch_size=8, max_length=512):
    
    if data_path == True:
        data_path = download_datasets(current_task, name_sub_info)
    
    checkpoints_path = f"checkpoints/{current_task}/{name_sub_info}/{run_name}/"
    name_to_save = f'{run_name}_{current_task}'+ '.pth'

    full_orig_target = []
    full_pred_labels = []

    model = T5ForConditionalGeneration.from_pretrained(model_identifier).to(device)
    model.load_state_dict(torch.load(checkpoints_path+name_to_save))
    test_dataset = getting_data(tokenizer, data_path, max_length=max_length, batch_size=batch_size, full_pipeline=False)

    for data in tqdm(test_dataset):
        predicted_token_ids = model.generate(input_ids=data['source_ids'].to(device), attention_mask=data['attention_mask'].to(device), max_length=128, do_sample=False, num_beams=1)

        lm_labels_hat = data['target_ids'].to(device).where(data['target_ids'].to(device)!=-100, torch.tensor(0, device=device))

        for i in range(len(predicted_token_ids)):
        
            target = tokenizer.decode(lm_labels_hat[i], skip_special_tokens=True)
            predicted = tokenizer.decode(predicted_token_ids[i], skip_special_tokens=True)

            new_origs_list, new_prediction_list = generate_labels(target, predicted)

            full_orig_target.extend(new_origs_list)
            full_pred_labels.extend(new_prediction_list)

    print('Scores:')
    acc_score = accuracy_score(full_orig_target, full_pred_labels)
    return acc_score
