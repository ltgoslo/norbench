#!/bin/env python3

from transformers import XLMRobertaTokenizer, XLMRobertaForTokenClassification, XLMRobertaTokenizerFast, DataCollatorForTokenClassification, Trainer, TrainingArguments
import argparse
import pandas as pd
import os
import glob
import numpy as np
import tensorflow as tf
from utils.ner_utils import models_type
from utils.model_utils import create_model, download_datasets
import random as python_random
import data_preparation.data_preparation_ner as data_preparation_ner
import evaluate_ner
from datasets import load_metric
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
metric = load_metric("seqeval")
import warnings
from conllu import parse
warnings.filterwarnings("ignore")

os.environ["TOKENIZERS_PARALLELISM"] = "true"
tagset = data_preparation_ner.tagset

# For reproducibility:
seed = 42
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)

def seq_ev_compute_metrics(p):
    "metrics that will be counted during evaluation"

    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    real_predictions = [
        [tagset[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [tagset[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=real_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def get_predictions(trainer, tokenized_data, tagset=tagset):
    "predictions"
    predictions, labels, _ = trainer.predict(tokenized_data["test"])
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    real_predictions = [
        [tagset[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return real_predictions


def model_init(model_name, task, tagset):
    
    model, tokenizer = create_model(model_name, task, len(tagset))
    
    return model, tokenizer


def init_args(output_dir, epochs=20):
    num_train_epochs = epochs
    per_device_train_batch_size = 2
    per_device_eval_batch_size = 8
    learning_rate = 3e-05
    weight_decay = 0.0
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    adam_epsilon = 1e-08
    max_grad_norm = 1.0
    num_warmup_steps = 750
    save_strategy = 'epoch'
    save_total_limit = 1
    load_best_model_at_end = True

    output_dir = output_dir
    overwrite_output_dir = False
    

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        evaluation_strategy="epoch",
        do_train=True,
        do_eval=True,
        do_predict=True,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon,
        max_grad_norm=max_grad_norm,
        num_train_epochs=num_train_epochs,
        warmup_steps=num_warmup_steps,
        load_best_model_at_end=load_best_model_at_end,
        seed=seed,
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
    )

    return training_args, output_dir


def initialization_trainer(model, tokenizer, tokenized_data, data_collator, output_dir, epochs, use_seqeval_evaluation, max_length=512, overwrite_cache=True, padding=False, label_all_tokens=False):
    
    set_seed(seed)

    training_args, output_dir = init_args(output_dir, epochs)

    # Initialize our Trainer
    if use_seqeval_evaluation == False:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["dev"],
            tokenizer=tokenizer,
            data_collator=data_collator,
        )
    else:
         trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_data["train"],
            eval_dataset=tokenized_data["dev"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=seq_ev_compute_metrics,
        )
    return trainer


def train_use_eval(data_path, sub_info, model_name, run_name, task, epochs, use_seqeval_evaluation, max_length=512, tagset=tagset):
    
    checkpoints_path = "checkpoints/" + task + '/' + sub_info + '/' + run_name + '/'

    # Load the dataset
    if data_path == True:
        data_path = download_datasets(task, sub_info)
        # checkpoints_path = "checkpoints/" + task + '/' + sub_info + '/' + run_name + '/'

    model, tokenizer = model_init(model_name, task, tagset)
    tokenized_data, data_collator = data_preparation_ner.collecting_data(tokenizer, data_path, max_length=max_length)

    trainer = initialization_trainer(model, tokenizer, tokenized_data, data_collator, checkpoints_path, epochs, use_seqeval_evaluation, max_length)
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
    trainer.state.save_to_json(
        os.path.join(checkpoints_path, "trainer_state.json")
    )
    #
    # Print Results
    output_train_file = os.path.join(checkpoints_path, "train_results.txt")
    with open(output_train_file, "w") as writer:
        print("**Train results**")
        for key, value in sorted(train_result.metrics.items()):
            print(f"{key} = {value}")
            writer.write(f"{key} = {value}\n")

    """# Evaluate the Model"""

    print("**Evaluate**")
    results = trainer.evaluate()

    output_eval_file = os.path.join(checkpoints_path, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        print("**Eval results**")
        for key, value in results.items():
            print(f"{key} = {value}")
            writer.write(f"{key} = {value}\n")
    return trainer, tokenized_data


def test(data_path, sub_info, model_name, task, run_name, trainer=None, tokenized_data=None, max_length=512, tagset=tagset):
    """# Run Predictions on the Test Dataset"""
    
    if data_path == True:
        data_path = download_datasets(task, sub_info)
        
    if trainer == None:
        _, tokenizer = model_init(model_name, task, tagset)
        
        checkpoints_path = "checkpoints/" + task + '/' + sub_info + '/' + run_name + '/'

        path_to_model = glob.glob(checkpoints_path + 'checkpoint-*/')[0]
        model = AutoModelForTokenClassification.from_pretrained(path_to_model)

        args, _ = init_args(checkpoints_path)

        tokenized_data, data_collator = data_preparation_ner.collecting_data(tokenizer, data_path, max_length, full_pipeline=False)
        
        trainer = Trainer(
                model=model,
                args=args,
                tokenizer=tokenizer,
                data_collator=data_collator,
            )
    
    print("**Predict**")
    real_predictions = get_predictions(trainer, tokenized_data)

    print('PREPARING TO SAVE PREDICTIONS')

    path_to_test = glob.glob(data_path + "/*{}.conllu".format('test'.split("_")[0]))[0]
    path_to_predictions = data_path+"predicted_{}.conllu".format(model_name.replace('/', '_'))

    test_conll = parse(open(path_to_test, "r").read())


    if len(test_conll) != len(real_predictions):
        raise ValueError("Check if there is enough spaces in the end of .conllu file")


    for nr, sentence in enumerate(test_conll):
        for tk_num, token in enumerate(sentence):
            token["misc"]["name"] = real_predictions[nr][tk_num]

    with open(path_to_predictions, "w") as f:
        for sentence in test_conll:
            f.write(sentence.serialize())
    print('Saving file')

    print('Scores:')
    test_results = evaluate_ner.evaluation(path_to_predictions, path_to_test)
    return test_results

