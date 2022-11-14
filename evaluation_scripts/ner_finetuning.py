#!/bin/env python3

from transformers import XLMRobertaTokenizer, XLMRobertaForTokenClassification, XLMRobertaTokenizerFast, DataCollatorForTokenClassification, Trainer, TrainingArguments
import argparse
import pandas as pd
import os
import glob
import numpy as np
from utils.ner_utils import models_type
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


def compute_metrics(p):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="auto")
    parser.add_argument("--model_name", default="ltgoslo/norbert")
    parser.add_argument("--run_model_name", default="norne_nob")
    parser.add_argument("--training_language", default="nob")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--use_seqeval_evaluation", type=bool, default=False)
    args = parser.parse_args()

    ner_data_path = "../data/ner/"

    model_type =  args.model_type 
    model_name = args.model_name
    dataset_name = args.run_model_name
    task_name = "ner"
    training_language = args.training_language
    lang_path = ner_data_path + training_language + "/"

    overwrite_cache = True
    seed = 42
    set_seed(seed)

    # Tokenizer
    padding = False
    max_length = 512
    label_all_tokens = False

    # Training
    num_train_epochs = args.epochs  # @param {type: "number"}
    per_device_train_batch_size = 16  # param {type: "integer"}
    per_device_eval_batch_size = 32  # param {type: "integer"}
    learning_rate = 3e-05  # @param {type: "number"}
    weight_decay = 0.0  # param {type: "number"}
    adam_beta1 = 0.9  # param {type: "number"}
    adam_beta2 = 0.999  # param {type: "number"}
    adam_epsilon = 1e-08  # param {type: "number"}
    max_grad_norm = 1.0  # param {type: "number"}
    num_warmup_steps = 750  # @param {type: "number"}
    save_strategy = 'epoch'
    save_total_limit = 1  # param {type: "integer"}
    load_best_model_at_end = True  # @param {type: "boolean"}

    output_dir = dataset_name + "_" + str(per_device_train_batch_size)
    overwrite_output_dir = False

    """# Initialize Training"""
    if model_type in models_type:
        if model_name in models_type[model_type]["model_names"].keys():
            model = models_type[model_type]["model"](models_type[model_type]["model_names"][model_name], num_labels=len(tagset))
            tokenizer = models_type[model_type]["tokenizer"](models_type[model_type]["model_names"][model_name])
        else:
            model = models_type[model_type]["model"](model_name, num_labels=len(tagset))
            tokenizer = models_type[model_type]["tokenizer"](model_name)
    else:
        model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=len(tagset))
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Load the dataset
    tokenized_data, data_collator = data_preparation_ner.collecting_data(tokenizer, lang_path)

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

    # Initialize our Trainer
    if args.use_seqeval_evaluation == False:
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
            compute_metrics=compute_metrics,
        )


    """# Start Training"""

    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
    trainer.state.save_to_json(
        os.path.join(training_args.output_dir, "trainer_state.json")
    )
    #
    # Print Results
    output_train_file = os.path.join(output_dir, "train_results.txt")
    with open(output_train_file, "w") as writer:
        print("**Train results**")
        for key, value in sorted(train_result.metrics.items()):
            print(f"{key} = {value}")
            writer.write(f"{key} = {value}\n")

    """# Evaluate the Model"""

    print("**Evaluate**")
    results = trainer.evaluate()

    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        print("**Eval results**")
        for key, value in results.items():
            print(f"{key} = {value}")
            writer.write(f"{key} = {value}\n")

    """# Run Predictions on the Test Dataset"""

    print("**Predict**")
    real_predictions = get_predictions(trainer, tokenized_data)

    # print(f"Scores on test dataset: {predictions}")

    print('PREPARING TO SAVE PREDICTIONS')

    path_to_test = glob.glob(lang_path + "/*{}.conllu".format('test'.split("_")[0]))[0]
    path_to_predictions = lang_path+"predicted_{}_{}.conllu".format(training_language, model_name.replace('/', '_'))

    test_conll = parse(open(path_to_test, "r").read())

    for nr, sentence in enumerate(test_conll):
        for tk_num, token in enumerate(sentence):
            token["misc"]["name"] = real_predictions[nr][tk_num]

    with open(path_to_predictions, "w") as f:
        for sentence in test_conll:
            f.write(sentence.serialize())
    print('Saving file')

    print('Scores:')
    test_results = evaluate_ner.evaluation(path_to_predictions, path_to_test)

    table = pd.DataFrame({"Train Lang": training_language,
                          "Test F1": [test_results]
                          })

    print(table)
    print(table.style.hide(axis='index').to_latex())
    table.to_csv("results/{}_ner.tsv".format(dataset_name.replace('/', '_')), sep="\t")

