from smart_open import open

import torch
from torch.utils.data import Dataset


class ColaDataset(Dataset):
    def __init__(self, path: str):
        self.sentences = []
        self.labels = []

        is_correct = False
        n_correct, n_incorrect = 0, 0
        for line in open(path):
            line = line.strip()
            if len(line) == 0 or line == "Ungrammatical:":
                continue
            if line == "Grammatical:":
                is_correct = True
                continue

            if is_correct: n_correct += 1
            else: n_incorrect += 1

            line = line.split('\t')[0]

            self.sentences.append(line)
            self.labels.append(1 if is_correct else 0)

        print(n_correct, n_incorrect, flush=True)

    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]

    def __len__(self):
        return len(self.sentences)


class CollateFunctor:
    def __init__(self, tokenizer, max_length): 
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, samples):
        sentences = [sentence for sentence, _ in samples]
        labels = [label for _, label in samples]

        encoding = self.tokenizer(
            sentences,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors='pt',
            padding=True
        )
        labels = torch.tensor(labels)

        return encoding.input_ids, encoding.attention_mask, labels
