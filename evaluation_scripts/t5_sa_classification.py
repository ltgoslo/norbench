#! /bin/env python3
# coding: utf-8

import argparse
import logging
import os
import random
import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn import metrics
from torch.optim import AdamW
from torch.utils import data
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def encoder(labels, texts, cur_tokenizer, cur_device):
    encoding = cur_tokenizer(
        texts,
        text_target=labels,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.maxl,
    ).to(cur_device)
    return encoding["labels"], encoding["input_ids"]


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
        help="Path to a T5 model",
        required=True,
    )
    arg(
        "--trainset",
        "-d",
        help="Path to a classification train set",
        required=True,
    )
    arg(
        "--devset",
        "-dev",
        help="Path to a classification dev set",
        required=True,
    )
    arg(
        "--testset",
        "-t",
        help="Path to a classification test set",
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

    if args.type == "sentence":
        logger.info("Fine-tuning for sentence-level sentiment analysis")
    elif args.type == "document":
        logger.info("Fine-tuning for document-level sentiment analysis")
        train_data, dev_data, test_data = (
            labels_6_to_3(train_data),
            labels_6_to_3(dev_data),
            labels_6_to_3(test_data),
        )

    num_classes = train_data["sentiment"].nunique()
    logger.info(f"We have {num_classes} classes")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(modelname, use_fast=False)

    mapping = {0: "negativ", 1: "nøytral", 2: "positiv"}
    label_ids = {el: tokenizer(mapping[el], add_special_tokens=False)["input_ids"][0]
                 for el in mapping}

    model = AutoModelForSeq2SeqLM.from_pretrained(modelname, trust_remote_code=True).to(device)

    model.train()

    optimizer = AdamW(model.parameters(), lr=1e-5)

    train_texts = train_data.review.to_list()
    text_labels = train_data.sentiment.to_list()
    text_labels = [mapping[el] for el in text_labels]

    dev_texts = dev_data.review.to_list()
    dev_labels = dev_data.sentiment.to_list()
    dev_labels = [mapping[el] for el in dev_labels]

    test_texts = test_data.review.to_list()
    test_labels = test_data.sentiment.to_list()
    test_labels = [mapping[el] for el in test_labels]

    logger.info(f"Tokenizing with max length {args.maxl}...")

    train_labels_tensor, train_encoding = encoder(
        text_labels, train_texts, tokenizer, device
    )
    test_labels_tensor, test_encoding = encoder(
        test_labels, test_texts, tokenizer, device
    )
    dev_labels_tensor, dev_encoding = encoder(
        dev_labels, dev_texts, tokenizer, device)

    logger.info("Tokenizing finished.")

    train_dataset = data.TensorDataset(train_encoding, train_labels_tensor)
    train_iter = data.DataLoader(train_dataset, batch_size=args.bsize, shuffle=True)

    dev_dataset = data.TensorDataset(dev_encoding, dev_labels_tensor)
    dev_iter = data.DataLoader(dev_dataset, batch_size=args.bsize, shuffle=False)

    test_dataset = data.TensorDataset(test_encoding, test_labels_tensor)
    test_iter = data.DataLoader(test_dataset, batch_size=args.bsize, shuffle=False)

    logger.info(f"Training with batch size {args.bsize} for {args.epochs} epochs...")

    fscores = []
    for epoch in range(args.epochs):
        losses = 0
        total_train_acc = 0
        all_predictions = []
        for text, label in tqdm.tqdm(train_iter):
            optimizer.zero_grad()
            outputs = model(input_ids=text, labels=label)
            loss = outputs.loss
            losses += loss.item()
            loss.backward()
            optimizer.step()
        train_loss = losses / len(train_iter)

        # Testing on the dev set:
        model.eval()
        dev_predictions = []
        dev_labels = []
        with torch.no_grad():
            for text, label in tqdm.tqdm(dev_iter):
                predictions = model.generate(
                    input_ids=text,
                    max_new_tokens=5,
                    return_dict_in_generate=True,
                    output_scores=True
                )

                decoded_labels = tokenizer.batch_decode(
                    label.cpu(), skip_special_tokens=True
                )

                marker_probabilities = [
                    [p[label_ids[0]], p[label_ids[1]], p[label_ids[2]]]
                    for p in predictions.scores[0].cpu()
                ]

                mapped_predictions = [
                    mapping[np.argmax(p)]
                    for p in marker_probabilities
                ]
                #predictions = tokenizer.batch_decode(
                #    predictions.sequences.cpu(), skip_special_tokens=True
                #)
                #for generated, marker, pred in zip(predictions[:2], marker_probabilities[:2], mapped_predictions[:2]):
                #    logger.info(f"{generated}\t{marker}\t{pred}")

                mapped_labels = [
                    mapping[0]
                    if mapping[0] in p
                    else mapping[2]
                    if mapping[2] in p
                    else mapping[1]
                    for p in decoded_labels
                ]
                dev_predictions += mapped_predictions
                dev_labels += mapped_labels
        precision, recall, fscore, support = metrics.precision_recall_fscore_support(
            dev_labels,
            dev_predictions,
            average="macro",
            zero_division=0,
        )
        logger.info(
            f"Epoch: {epoch}, Train loss: {train_loss:.4f}, Dev F1: {fscore:.4f}"
        )
        fscores.append(fscore)
        if len(fscores) > 2:
            if fscores[-1] < fscores[-2]:
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
        for text, label in tqdm.tqdm(test_iter):
            predictions = model.generate(
                input_ids=text,
                max_new_tokens=5,
                return_dict_in_generate=True,
                output_scores=True
            )
            # predictions = tokenizer.batch_decode(
            #     predictions.cpu(), skip_special_tokens=True
            # )
            decoded_labels = tokenizer.batch_decode(
                label.cpu(), skip_special_tokens=True
            )
            mapped_predictions = [
                [p[label_ids[0]], p[label_ids[1]], p[label_ids[2]]]
                for p in predictions.scores[0].cpu()
            ]

            mapped_predictions = [
                mapping[np.argmax(p)]
                for p in mapped_predictions
            ]

            mapped_labels = [
                mapping[0]
                if mapping[0] in p
                else mapping[2]
                if mapping[2] in p
                else mapping[1]
                for p in decoded_labels
            ]
            test_predictions += mapped_predictions
            test_labels += mapped_labels

    precision, recall, fscore, support = metrics.precision_recall_fscore_support(
        test_labels, test_predictions, average="macro", zero_division=0
    )
    scores.append(fscore)
    logger.info(
        metrics.classification_report(test_labels, test_predictions, zero_division=0)
    )

    with open("scores/" + current_name + ".tsv", "a") as f:
        f.write(f"{args.seed}\t{scores[0]}\t{scores[1]}\n")
