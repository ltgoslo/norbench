import argparse
import torch
import random
import math
from collections import defaultdict
from statistics import mean, stdev
from tqdm import tqdm
import torchmetrics
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  #, GenerationConfig
from modeling_nort5_cache import NorT5ForConditionalGeneration

from dataset import Dataset, CollateFunctor



class StringExactMatchScore(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target):
        assert len(preds) == len(target)
        self.correct += sum(1 if p == t else 0 for p, t in zip(preds, target))
        self.total += len(preds)

    def compute(self):
        return self.correct.float() / self.total


class BleuScore(torchmetrics.SacreBLEUScore):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, preds, target):
        super().update(preds, [[t] for t in target])

    def compute(self):
        return super().compute()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="../final_models/nort5-base", type=str)
    parser.add_argument("--lr", default=2.0e-5, type=float, help="BERT learning rate.")
    parser.add_argument("--weight_decay", default=0.1, type=float, help="BERT learning rate.")
    parser.add_argument("--warmup_portion", default=0.05, type=float, help="BERT learning rate.")
    parser.add_argument("--max_length", default=128, type=int, help="BERT learning rate.")
    parser.add_argument("--acummulation_steps", default=1, type=int)
    parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument('--mixed_precision', default=False, action=argparse.BooleanOptionalAction)

    args = parser.parse_args()

    return args


def setup_training(seed, args):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda")
    return device


if __name__ == "__main__":
    args = parse_arguments()

    seed_results = defaultdict(dict)
    for seed in [1234, 2345, 3456, 4567, 5678]:
        device = setup_training(seed, args)

        tokenizer = AutoTokenizer.from_pretrained(args.model)

        if "old" in args.model:
            model = AutoModelForSeq2SeqLM.from_pretrained(args.model).to(device)
        else:
            model = NorT5ForConditionalGeneration.from_pretrained(args.model).to(device)

        train_set = Dataset("nb_nn_train.tsv.gz", limit=True)
        valid_set = Dataset("nb_nn_dev.tsv.gz", limit=True)
        test_set = Dataset("nb_nn_test.tsv.gz", limit=True)

        metrics = {
            "BLEU": BleuScore(),
            "EM": StringExactMatchScore()
        }

        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size // args.acummulation_steps,
            shuffle=True,
            drop_last=True,
            collate_fn=CollateFunctor(tokenizer, args.max_length),
            num_workers=4,
            pin_memory=True
        )
        valid_loader = DataLoader(
            valid_set,
            batch_size=args.batch_size // args.acummulation_steps,
            shuffle=False,
            drop_last=False,
            collate_fn=CollateFunctor(tokenizer, args.max_length),
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_set,
            batch_size=args.batch_size // args.acummulation_steps,
            shuffle=False,
            drop_last=False,
            collate_fn=CollateFunctor(tokenizer, args.max_length),
            num_workers=4,
            pin_memory=True
        )

        no_decay = ['bias', "layer_norm", "embedding", "LayerNorm", "Embedding"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
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

        scheduler = cosine_schedule_with_warmup(optimizer, args.epochs*len(train_loader) * args.warmup_portion, args.epochs*len(train_loader), 0.1)

        grad_scaler = torch.cuda.amp.GradScaler(enabled=args.mixed_precision)
        best_bleu = 0.0

        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            for i, batch in enumerate(tqdm(train_loader)):
                source_ids, attention_mask, target_ids = (item.to(device) for item in batch)

                with torch.cuda.amp.autocast(args.mixed_precision):
                    loss = model(
                        input_ids=source_ids,
                        attention_mask=attention_mask,
                        labels=target_ids
                    ).loss

                grad_scaler.scale(loss / args.acummulation_steps).backward()


                if (i + 1) % args.acummulation_steps == 0:
                    grad_scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=25.0)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)


            model.eval()
            for metric in metrics.values():
                metric.reset()
            with torch.no_grad():
                results = {}
                for i, batch in enumerate(tqdm(valid_loader)):
                    optimizer.zero_grad(set_to_none=True)
                    source_ids, attention_mask, target_ids = (item.to(device) for item in batch)

                    with torch.cuda.amp.autocast(args.mixed_precision):
                        predictions = model.generate(
                            input_ids=source_ids,
                            attention_mask=attention_mask,
                            do_sample = False, # dette forstår jeg ikke ennå 
                            max_new_tokens = 128, 
                            top_k = 0, 
                        )
                        sources = tokenizer.batch_decode(source_ids.cpu(), skip_special_tokens=True)
                        predictions = tokenizer.batch_decode(predictions.cpu(), skip_special_tokens=True)
                        targets = tokenizer.batch_decode(target_ids.cpu(), skip_special_tokens=True)

                        if i == 0:
                            for s, p, t in zip(sources, predictions, targets):
                                print(f"SOURCE:{s}\nGOLD:  {t}\nPRED:  {p}\n", flush=True)
                        for metric in metrics.values():
                            metric.update(
                                predictions,
                                targets
                            )

                for metric_name, metric in metrics.items():
                    results[f"valid/{metric_name}"] = metric.compute().item() * 100.0
                    print(f"$$$ {epoch}\t{metric.compute().item() * 100.0}", flush=True)


            print(results, flush=True)

            if results["valid/BLEU"] <= best_bleu:
                continue
            best_bleu = results["valid/BLEU"]


            model.eval()
            for metric in metrics.values():
                metric.reset()
            with torch.no_grad():
                results = {}
                for i, batch in enumerate(tqdm(test_loader)):
                    optimizer.zero_grad(set_to_none=True)
                    source_ids, attention_mask, target_ids = (item.to(device) for item in batch)

                    with torch.cuda.amp.autocast(args.mixed_precision):
                        predictions = model.generate(
                            input_ids=source_ids,
                            attention_mask=attention_mask,
                        )
                        sources = tokenizer.batch_decode(source_ids.cpu(), skip_special_tokens=True)
                        predictions = tokenizer.batch_decode(predictions.cpu(), skip_special_tokens=True)
                        targets = tokenizer.batch_decode(target_ids.cpu(), skip_special_tokens=True)

                        if i == 0:
                            for s, p, t in zip(sources, predictions, targets):
                                print(f"SOURCE:{s}\nGOLD:  {t}\nPRED:  {p}\n", flush=True)
                        for metric in metrics.values():
                            metric.update(
                                predictions,
                                targets
                            )

        for metric_name, metric in metrics.items():
            results[f"valid/{metric_name}"] = metric.compute().item() * 100.0

        print(results, flush=True)

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
