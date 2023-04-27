# Based on seq_label.py
# Loads the json file made in helpers.ipynb with best epoch for all experiments
# Runs the training and evaluation script without any saving of model
# Records the predict_f1 for each seed
# Arguement: A json file with the training parameters that are not overridden here. Use one for the initial training.

# Model iteration is done here, so that this script should only be called once for all the experiments. The script reads the  test_setup.json where all models with their best epoch are stored.


# %%
from datasets import ClassLabel,  load_dataset, load_from_disk, DatasetDict, Dataset
import os, sys, json
from pathlib import Path
import evaluate
import transformers
import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)

from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from src.local_parsers import ModelArguments, DataTrainingArguments
from modeling_norbert import NorbertForTokenClassification
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["WANDB_DISABLED"] = "true"
print("Numpy:", np.version.version)
print("PyTorch:", torch.__version__)
print("Transformers:", transformers.__version__)

# EXP_DATA_PATH = "configs/evals/test_setup.json"
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
# test_data = json.loads(Path(EXP_DATA_PATH).read_text())

too_large = ["NorBERT_3_large", "XLM-R_large"] # Had to reduce batch size for these.
# Parse from json file submitted as argument to the .py file
if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    # If we pass only one argument to the script and it's the path to a json file,

    with open (os.path.abspath(sys.argv[1])) as rf:
        config_json = json.load(rf)
        args_dict = config_json["args_dict"]
else:
    print(sys.argv[0], "Made for getting one parameter, the json config file")
    sys.exit()
if config_json["best_epoch"] is None:
    print(sys.argv[1], "No best epoch registered for this config. Exiting.")
    sys.exit()

label_col = args_dict["label_column_name"]

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    if data_args.return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

def get_label_list(labels):
    unique_labels = set()
    for label in labels:
        unique_labels = unique_labels | set(label)
    label_list = list(unique_labels)
    sorted_labels = sorted(label_list,key=lambda name: (name[1:], name[0])) # Gather B and I
    return sorted_labels

# Tokenize all texts and align the labels with them.
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples[text_column_name],
        padding=padding,
        truncation=True,
        max_length=data_args.max_seq_length,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples[label_column_name]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None or word_idx == previous_word_idx :
                label_ids.append(-100)
            # We set the label for the first token of each word only.
            else : #New word
                label_ids.append(label_to_id[label[word_idx]])
            # We do not keep the option to label the subsequent subword tokens here.

            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs



if args_dict["task_name"] in too_large:
    args_dict["per_device_train_batch_size"] = int(args_dict["per_device_train_batch_size"]) // 2




# Here we override training arguments from the json file
args_dict["num_train_epochs"] = config_json["best_epoch"]
args_dict["do_eval"] = False
args_dict["logging_strategy"] =  "no"
args_dict["evaluation_strategy"] =  "no"
args_dict["save_strategy"] =  "no"
args_dict[ "save_total_limit"]= 1
args_dict["load_best_model_at_end"]= False
args_dict["do_predict"] = False

model_args, data_args, training_args = parser.parse_dict(args_dict)

text_column_name = data_args.text_column_name
label_column_name = data_args.label_column_name
assert data_args.label_all_tokens == False, "Our script only labels first subword token"
dsd = load_from_disk(data_args.dataset_name)
transformers.logging.set_verbosity_warning()
# %%

label_list = get_label_list(dsd["train"][data_args.label_column_name]) # "tsa_tags"
label_to_id = {l: i for i, l in enumerate(label_list)}
num_labels = len(label_list)
labels_are_int = False
label_list

# %%
config = AutoConfig.from_pretrained(
    model_args.config_name if model_args.config_name else model_args.model_name_or_path,
    num_labels=num_labels,
    finetuning_task=data_args.task_name,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)
tokenizer_name_or_path = model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path
if config.model_type in {"gpt2", "roberta"}:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        add_prefix_space=True,
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

# %%
# Instanciate the model
if "norbert3" in model_args.model_name_or_path:

    model = NorbertForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )


else:
        model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
    )


# %%

# print("Model's label2id:", model.config.label2id)
print("Our label2id:    ", label_to_id)
# print("Our label list:  ", label_list)
# print("PretrainedConfig", PretrainedConfig(num_labels=num_labels).label2id)
assert (model.config.label2id == PretrainedConfig(num_labels=num_labels).label2id) or (model.config.label2id == label_to_id), "Model seems to have been fine-tuned on other labels already. Our script does not adapt to that."


# Set the correspondences label/ID inside the model config
model.config.label2id = {l: i for i, l in enumerate(label_list)}
model.config.id2label = {i: l for i, l in enumerate(label_list)}


# Preprocessing the dataset
# Padding strategy
padding = "max_length" if data_args.pad_to_max_length else False


# %%
with training_args.main_process_first(desc="train dataset map pre-processing"):
    train_dataset = dsd["train"].map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file= False,
        desc="Running tokenizer on train dataset",
    )
with training_args.main_process_first(desc="validation dataset map pre-processing"):
    eval_dataset = dsd["validation"].map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on validation dataset",
    )
with training_args.main_process_first(desc="validation dataset map pre-processing"):
    predict_dataset = dsd["test"].map(
        tokenize_and_align_labels,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on test dataset",
    )


# %%
# import seqeval
data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)

# Metrics
metric = evaluate.load("seqeval") # 
# metric = evaluate.evaluator(task = 'token-classification' )

# %%
num_seeds = config_json.get("num_seeds", 5)
seeds = [(n+1)*101 for n in range(num_seeds)]
print("\n Ready to train and evaluate", config_json["model_shortname"], "with seeds", seeds)
pred_f1s = {}
for seed in seeds:
    training_args.seed = seed
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )


    train_result = trainer.train(resume_from_checkpoint=False)
    metrics = train_result.metrics
    metrics["train_samples"] =  len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # Evaluate
    print("\nEvaluation,",model_args.model_name_or_path)

    print()

    # %%
    # Predict
    from src.evaluate_tsa import evaluateur
    # import pickle

    # Debug
    # predict_dataset = predict_dataset.select([999])

    trainer_predict = trainer.predict(predict_dataset, metric_key_prefix="predict")
    predictions, labels, metrics = trainer_predict




    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # trainer.log_metrics("predict", metrics)
    # trainer.save_metrics("predict", metrics)
    gold = predict_dataset[args_dict["label_column_name"]] # Note: Will not work if dataset has ints, and the text labels in metadata
    for g, pred in zip(gold,true_predictions ):
        assert len(g) == len(pred), (len(g) , len(pred))

    batista_f1 = evaluateur(gold, true_predictions)
    # try:
    #     if data_args.return_entity_level_metrics:
    #         seqeval_f1 =  metrics["predict_overall_f1"]
    #     else:
    #         seqeval_f1 =  metrics["predict_f1"]
    # except:
    #     seqeval_f1 = 0
    print("batista_f1",batista_f1)
    # print("seqeval_f1",seqeval_f1)
    # print("Seqeval > Batista:",round(seqeval_f1 - batista_f1, 4))
    # metrics["seqeval_f1"] = seqeval_f1
    # metrics["batista_f1"] = batista_f1
    pred_f1s[seed] = batista_f1

pred_vals = np.array(list(pred_f1s.values()))
pred_f1s["mean"] = np.mean(pred_vals)
pred_f1s["std"] = np.std(pred_vals)
print("Evaluation results:",pred_f1s )
config_json["tests"] = pred_f1s
with open (os.path.abspath(sys.argv[1]), "w") as wf:
    json.dump(config_json, wf)

