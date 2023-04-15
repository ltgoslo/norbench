#! /bin/env python3
# coding: utf-8

import pandas as pd
import torch
from torch.utils import data
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import argparse
import logging
import tqdm
from sklearn import metrics
import os
import random
import numpy as np
import sys


def multi_acc(y_pred, y_test):
    batch_predictions = torch.log_softmax(y_pred, dim=1).argmax(dim=1)
    correctness = batch_predictions == y_test
    acc = torch.sum(correctness).item() / y_test.size(0)
    return acc, batch_predictions.tolist()


def encoder(labels, texts, cur_tokenizer, cur_device):
    labels_tensor = torch.tensor(labels).to(cur_device)
    encoding = cur_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.maxl,
    ).to(cur_device)
    return labels_tensor, encoding


def labels_6_to_3(df):
    df.sentiment = df.sentiment.replace(1, 0)
    df.sentiment = df.sentiment.replace(2, 0)
    df.sentiment = df.sentiment.replace(3, 1)
    df.sentiment = df.sentiment.replace(4, 2)
    df.sentiment = df.sentiment.replace(5, 2)

    return df


def seed_everything(seed_value=42):
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    return seed_value


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
    )
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg(
        "--model",
        "-m",
        help="Path to a BERT model",
        required=True,
    )
    arg(
        "--trainset",
        "-d",
        help="Path to a sentence classification train set",
        required=True,
    )
    arg(
        "--devset",
        "-dev",
        help="Path to a sentence classification dev set",
        required=True,
    )
    arg(
        "--testset",
        "-t",
        help="Path to a sentence classification test set",
        required=True,
    )
    arg(
        "--type",
        choices=["sentence", "document"],
        help="Sentence or document classification",
        default="sentence",
    )
    arg("--epochs", "-e", type=int, help="Number of epochs", default=10)
    arg("--maxl", "-l", type=int, help="Max length", default=512)
    arg("--bsize", "-b", type=int, help="Batch size", default=16)
    arg("--seed", "-s", type=int, help="Random seed", default=42)
    arg("--identifier", "-i", help="Model identifier", default="model")
    arg("--freeze", "-f", action="store_true", help="Freeze the model?")
    arg("--custom", action="store_true", help="Custom wrapper?")
    arg("--save", help="Where to save the finetuned model")

    args = parser.parse_args()

    modelname = args.model
    dataset = args.trainset
    devset = args.devset
    testset = args.testset

    _ = seed_everything(args.seed)
    logger.info(f"Training with seed {args.seed}...")

    current_name = "_".join([args.identifier, args.type])

    logger.info("Reading train data...")
    train_data = pd.read_csv(dataset)
    logger.info("Train data reading complete.")

    logger.info("Reading dev data...")
    dev_data = pd.read_csv(devset)
    logger.info("Dev data reading complete.")

    logger.info("Reading test data...")
    test_data = pd.read_csv(testset)
    logger.info("Test data reading complete.")

    mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}

    if args.type == "sentence":
        logger.info("Fine-tuning for sentence-level sentiment analysis")
    elif args.type == "document":
        logger.info("Fine-tuning for document-level sentiment analysis")
        mapping = {0: "Negative", 1: "Fair", 2: "Positive"}
        train_data, dev_data, test_data = (
            labels_6_to_3(train_data),
            labels_6_to_3(dev_data),
            labels_6_to_3(test_data),
        )

    num_classes = train_data["sentiment"].nunique()
    logger.info(f"We have {num_classes} classes")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(modelname, use_fast=False)

    if args.custom:
        logger.info('You are using a custom wrapper, NOT a HuggingFace model.')
        sys.path.append(modelname)
        from modeling_norbert import NorbertForSequenceClassification
        model = NorbertForSequenceClassification.from_pretrained(
            modelname, num_labels=num_classes).to(device)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            modelname, num_labels=num_classes
        ).to(device)

    if args.freeze:
        logger.info("Freezing the model, training only the classifier on top")
        for param in model.base_model.parameters():
            param.requires_grad = False

    model.train()

    optimizer = AdamW(model.parameters(), lr=1e-5)

    train_texts = train_data.review.to_list()
    text_labels = train_data.sentiment.to_list()

    dev_texts = dev_data.review.to_list()
    dev_labels = dev_data.sentiment.to_list()

    test_texts = test_data.review.to_list()
    test_labels = test_data.sentiment.to_list()

    logger.info(f"Tokenizing with max length {args.maxl}...")

    train_labels_tensor, train_encoding = encoder(
        text_labels, train_texts, tokenizer, device
    )
    test_labels_tensor, test_encoding = encoder(
        test_labels, test_texts, tokenizer, device
    )
    dev_labels_tensor, dev_encoding = encoder(dev_labels, dev_texts, tokenizer, device)

    input_ids = train_encoding["input_ids"]
    attention_mask = train_encoding["attention_mask"]
    dev_input_ids = dev_encoding["input_ids"]
    dev_attention_mask = dev_encoding["attention_mask"]
    test_input_ids = test_encoding["input_ids"]
    test_attention_mask = test_encoding["attention_mask"]
    logger.info("Tokenizing finished.")

    train_dataset = data.TensorDataset(input_ids, attention_mask, train_labels_tensor)
    train_iter = data.DataLoader(train_dataset, batch_size=args.bsize, shuffle=True)

    dev_dataset = data.TensorDataset(
        dev_input_ids, dev_attention_mask, dev_labels_tensor
    )
    dev_iter = data.DataLoader(dev_dataset, batch_size=args.bsize, shuffle=False)

    test_dataset = data.TensorDataset(
        test_input_ids, test_attention_mask, test_labels_tensor
    )
    test_iter = data.DataLoader(test_dataset, batch_size=args.bsize, shuffle=False)

    logger.info(f"Training with batch size {args.bsize} for {args.epochs} epochs...")

    fscores = []
    for epoch in range(args.epochs):
        losses = 0
        total_train_acc = 0
        all_predictions = []
        for text, mask, label in tqdm.tqdm(train_iter):
            optimizer.zero_grad()
            outputs = model(text, attention_mask=mask, labels=label)
            loss = outputs.loss
            losses += loss.item()
            predictions = outputs.logits
            accuracy, predicted_labels = multi_acc(predictions, label)
            all_predictions += predicted_labels
            total_train_acc += accuracy
            loss.backward()
            optimizer.step()
        train_acc = total_train_acc / len(train_iter)
        train_loss = losses / len(train_iter)

        # Testing on the dev set:
        model.eval()
        dev_predictions = []
        dev_labels = []
        with torch.no_grad():
            for text, mask, label in tqdm.tqdm(dev_iter):
                outputs = model(text, attention_mask=mask)
                predictions = outputs.logits
                accuracy, predicted_labels = multi_acc(predictions, label)
                dev_predictions += predicted_labels
                dev_labels += label.tolist()
        precision, recall, fscore, support = metrics.precision_recall_fscore_support(
            [mapping[el] for el in dev_labels],
            [mapping[el] for el in dev_predictions],
            average="macro",
            zero_division=0,
        )
        logger.info(
            f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f}, "
            f"Dev F1: {fscore:.4f}"
        )
        fscores.append(fscore)
        if len(fscores) > 2:
            if fscores[-1] <= fscores[-2]:
                logger.info("Early stopping!")
                break
        model.train()

    # Final testing on the test set
    scores = [fscores[-1]]
    model.eval()

    logger.info(f"Testing on the test set with batch size {args.bsize}...")

    test_predictions = []
    test_labels = []
    with torch.no_grad():
        for text, mask, label in tqdm.tqdm(test_iter):
            outputs = model(text, attention_mask=mask)
            predictions = outputs.logits
            accuracy, predicted_labels = multi_acc(predictions, label)
            test_predictions += predicted_labels
            test_labels += label.tolist()

    mapped_labels = [mapping[el] for el in test_labels]
    mapped_predictions = [mapping[el] for el in test_predictions]
    precision, recall, fscore, support = metrics.precision_recall_fscore_support(
        mapped_labels, mapped_predictions, average="macro", zero_division=0
    )
    scores.append(fscore)
    logger.info(
        metrics.classification_report(
            mapped_labels, mapped_predictions, zero_division=0
        )
    )

    with open("scores/" + current_name + ".tsv", "a") as f:
        f.write(f"{args.seed}\t{scores[0]}\t{scores[1]}\n")
