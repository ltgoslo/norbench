from smart_open import open
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, path: str):
        self.nb = []
        self.nn = []

        for i, line in enumerate(open(path)):
            line = line.strip()
            try:
                nb, nn = line.split('\t')
            except:
                print(f"No translation pair, skipping: {line}", flush=True)

            self.nb.append(nb)
            self.nn.append(nn)

    def __getitem__(self, index):
        return self.nb[index], self.nn[index]

    def __len__(self):
        return len(self.nb)


class CollateFunctor:
    def __init__(self, tokenizer, max_length): 
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, samples):
        nb = [nb for nb, _ in samples]
        nn = [nn for _, nn in samples]

        nb = self.tokenizer(
            nb,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
            padding=True
        )
        nn = self.tokenizer(
            nn,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True,
            padding=True
        )

        return nb.input_ids, nb.attention_mask, nn.input_ids
