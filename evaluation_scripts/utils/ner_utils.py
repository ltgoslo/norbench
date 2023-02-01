#!/bin/env python3

# the code has been partly taken from https://huggingface.co/docs/transformers/model_doc/t5 Token Classification tutorial

def get_ner_tags():
    return ['O', 'B-PER', 'B-LOC',  'I-PER',  'B-PROD',  'B-GPE_LOC',  'I-PROD', 
            'B-DRV',  'I-DRV',  'B-EVT',  'I-EVT',  'B-ORG',  'I-LOC',  'I-GPE_LOC',
            'I-ORG',  'B-GPE_ORG',  'I-GPE_ORG',  'B-MISC', 'I-MISC']
        

def organized_subsets(data, id2label):
    ids, tokens, tags = [], [], []
    for el in data:
        ids.append(el['id'])
        tokens.append(el['tokens'])
        tags.append([id2label[tag] for tag in el['ner_tags']])
    return ids, tokens, tags


def organized_subset_t5(data, id2label):
    ids, tokens, tags, tags_indexed = [], [], [], []
    for el in data:
        ids.append(el['id'])
        tokens.append(el['tokens'])
        tags.append(el['ner_tags'])
        tags_indexed.append([id2label[tag] for tag in el['ner_tags']])
    return ids, tokens, tags, tags_indexed


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def find_sub_list(sl, l):
    results = []
    sll = len(sl)
    for ind in (i for i, e in enumerate(l) if e == sl[0]):
        if l[ind:ind+sll] == sl:
            results.append((ind, ind+sll-1))
    return results


def generate_label(input: str, target: str):

    mapper = { 'O': 0, 'B-DRV': 1, 'I-DRV': 2, 'B-EVT': 3,  'I-EVT': 4,
    'B-GPE_LOC': 5, 'I-GPE_LOC': 6, 'B-GPE_ORG': 7,  'I-GPE_ORG': 8,
    'B-LOC': 9, 'I-LOC': 10,  'B-MISC': 11, 'I-MISC': 12, 'B-PER': 13, 'I-PER': 14,
    'B-ORG': 15,  'I-ORG': 16, 'B-PROD': 17, 'I-PROD': 18}
    
    inv_mapper = {v: k for k, v in mapper.items()}

    input = input.split(" ")
    target = target.split("; ")

    init_target_label = [mapper['O']]*len(input)

    for ent in target:
        ent = ent.split(": ")
        try:
            sent_end = ent[1].split(" ")
            index = find_sub_list(sent_end, input)
        except:
            continue
        # print(index)
        try:
            init_target_label[index[0][0]] = mapper[f"B-{ent[0].upper()}"]
            for i in range(index[0][0]+1, index[0][1]+1):
                init_target_label[i] = mapper[f"I-{ent[0].upper()}"]
        except:
            continue
    init_target_label = [inv_mapper[j] for j in init_target_label]
    return init_target_label