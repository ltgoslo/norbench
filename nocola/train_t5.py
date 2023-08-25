import argparse
import torch
import random
import math
from collections import defaultdict
from statistics import mean, stdev
from tqdm import tqdm
import torchmetrics
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from dataset_t5 import ColaDataset, CollateFunctor


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="ltg/nort5-base", type=str)
    parser.add_argument("--task", default="cola", type=str, help="GLUE task.")
    parser.add_argument("--lr", default=5.0e-6, type=float, help="BERT learning rate.")
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument('--mixed_precision', default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    return args


def setup_training(seed):
    assert torch.cuda.is_available()

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")
    return device


if __name__ == "__main__":
    args = parse_arguments()

    seed_results = defaultdict(dict)
    for seed in [1234, 2345, 3456, 4567, 5678]:
        device = setup_training(seed)

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForSeq2SeqLM.from_pretrained(args.model, trust_remote_code=True).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{n_params}", flush=True)

        train_set = ColaDataset("./data/NoCoLa_class_train.txt")
        valid_set = ColaDataset("./data/NoCoLa_class_dev.txt")
        test_set = ColaDataset("./data/NoCoLa_class_test.txt")

        if args.task == "cola":
            metrics = {
                "accuracy": torchmetrics.Accuracy(),
                "f1": torchmetrics.F1Score(num_classes=2, multiclass=False),
                "MCC": torchmetrics.MatthewsCorrCoef(num_classes=2)
            }

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=CollateFunctor(tokenizer, 512),
            num_workers=4,
            pin_memory=True
        )
        valid_loader = DataLoader(
            valid_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=CollateFunctor(tokenizer, 512),
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            collate_fn=CollateFunctor(tokenizer, 512),
            num_workers=4,
            pin_memory=True
        )

        no_decay = ['bias', "layer_norm", "embedding", "LayerNorm", "Embedding"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
                "lr": args.lr
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
                "lr": args.lr
            }
        ]

        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, eps=1e-6)

        def cosine_schedule_with_warmup(optimizer, num_warmup_steps: int, num_training_steps: int, min_factor: float):
            def lr_lambda(current_step):
                if current_step < num_warmup_steps:
                    return float(current_step) / float(max(1, num_warmup_steps))
                progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
                lr = max(min_factor, min_factor + (1 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * progress)))

                return lr

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        scheduler = cosine_schedule_with_warmup(optimizer, args.epochs*len(train_loader) * 0.06, args.epochs*len(train_loader), 0.1)

        grad_scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)
        best_mcc = -100

        for epoch in range(args.epochs):
            model.train()
            for batch in tqdm(train_loader):
                optimizer.zero_grad(set_to_none=True)
                input_ids, attention_mask, labels = (item.to(device) for item in batch)

                with torch.cuda.amp.autocast(args.mixed_precision):
                    loss = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels[:, 1:]
                    ).loss

                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                scheduler.step()

            print(torch.cuda.max_memory_allocated(), flush=True)

            model.eval()
            for metric in metrics.values():
                metric.reset()
            with torch.no_grad():
                results = {}
                for batch in tqdm(valid_loader):
                    optimizer.zero_grad(set_to_none=True)
                    input_ids, attention_mask, labels = (item.to(device) for item in batch)

                    with torch.cuda.amp.autocast(args.mixed_precision):
                        predictions = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=5,
                        )
                        predictions = tokenizer.batch_decode(predictions.cpu(), skip_special_tokens=True)
                        labels = tokenizer.batch_decode(labels.cpu(), skip_special_tokens=True)

                        fake_predictions = [[1.0, -1.0] if "uriktig" in p else [-1.0, 1.0] for p in predictions]
                        fake_labels = [0 if "uriktig" in l else 1 for l in labels]

                        for metric in metrics.values():
                            metric.update(
                                torch.tensor(fake_predictions),
                                torch.tensor(fake_labels)
                            )

                for metric_name, metric in metrics.items():
                    results[f"valid/{metric_name}"] = metric.compute().item() * 100.0
                    print(f"$$$ {epoch}\t{metric.compute().item() * 100.0}", flush=True)

                print(results, flush=True)
            
                if results["valid/MCC"] <= best_mcc:
                    continue
                best_mcc = results["valid/MCC"]

                for metric in metrics.values():
                    metric.reset()
                for batch in tqdm(test_loader):
                    optimizer.zero_grad(set_to_none=True)
                    input_ids, attention_mask, labels = (item.to(device) for item in batch)

                    with torch.cuda.amp.autocast(args.mixed_precision):
                        predictions = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=5,
                        )
                        predictions = tokenizer.batch_decode(predictions.cpu(), skip_special_tokens=True)
                        labels = tokenizer.batch_decode(labels.cpu(), skip_special_tokens=True)

                        fake_predictions = [[1.0, -1.0] if "uriktig" in p else [-1.0, 1.0] for p in predictions]
                        fake_labels = [0 if "uriktig" in l else 1 for l in labels]

                        for metric in metrics.values():
                            metric.update(
                                torch.tensor(fake_predictions),
                                torch.tensor(fake_labels)
                            )

                for metric_name, metric in metrics.items():
                    print(metric_name, metric.compute().item() * 100.0, flush=True)
                    seed_results[metric_name][seed] = metric.compute().item() * 100.0

    r = {key: f"{mean(seeds.values()):.2f}$^{{\\pm{stdev(seeds.values()):.2f}}}$" for key, seeds in seed_results.items()}
    print(args.model)
    print(' & '.join(r.keys()))
    print(' & '.join(r.values()), flush=True)

    with open(f"results_{args.model.split('/')[-1]}.txt", 'a') as f:
        f.write(' & '.join(r.keys()) + '\n')
        f.write(' & '.join(r.values()) + '\n')
