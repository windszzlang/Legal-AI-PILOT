import json
import torch
import os
import random
import numpy as np
import json
import re
import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, jaccard_score, average_precision_score, hamming_loss, roc_auc_score



def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def load_labels(label_path):
    with open(label_path) as f:
        ECHR_info = json.load(f)
    labels = ECHR_info['contents'].keys()
    laws = []
    for v in ECHR_info['contents'].values():
        laws.append(v['title'] + ': ' +  v['text'])
    return labels, laws


def load_data(data_path, tokenizer=None, load_part=None):
    data = []
    max_len = 0
    with open(data_path) as f:
        for i, line in enumerate(f.readlines()):
            # if i + 1 == 589 or i + 1 == 1123 or i + 1 == 7601 or i + 1 == 7874 or i + 1 == 8751:
                # continue

            if load_part != None and i + 1 > load_part:
                break
            D = json.loads(line)
            data.append(D)
    return data


def save_data(data, data_path):
    with open(data_path, 'w') as f:
        for D in data:
            f.write(json.dumps(D) + '\n')



def get_label_id_map(labels):
    id2label = dict(enumerate(labels))
    label2id = {label: id for id, label in id2label.items()}
    label_num = len(labels)
    return id2label, label2id, label_num

def pred2label(pred, id2label):
    labels = []
    for i, p in enumerate(pred):
        if p == 1:
            labels.append(id2label[i])
    return labels


def compute_violate_performance(pred, gold):
    pred = [1 if p.sum() else 0 for p in pred]
    gold = [1 if g.sum() else 0 for g in gold]
    scores = {}
    scores['f1_score'] = f1_score(gold, pred, average='macro')
    scores['precision_score'] = precision_score(gold, pred, average='macro')
    scores['recall_score'] = recall_score(gold, pred, average='macro')
    scores['cls_report'] = classification_report(gold, pred)
    return scores


def compute_article_performance(pred, gold):
    pred = [p.cpu().tolist() for p in pred]
    gold = [g.cpu().tolist() for g in gold]
    scores = {}
    scores['macro_f1_score'] = f1_score(gold, pred, average='macro')
    scores['micro_f1_score'] = f1_score(gold, pred, average='micro')
    scores['precision_score'] = precision_score(gold, pred, average='micro')
    scores['recall_score'] = recall_score(gold, pred, average='micro')
    scores['macro_jaccard_score'] = jaccard_score(gold, pred, average='macro')
    scores['micro_jaccard_score'] = jaccard_score(gold, pred, average='micro')
    scores['macro_PRAUC'] = average_precision_score(gold, pred, average='macro') # PRAUC
    scores['micro_PRAUC'] = average_precision_score(gold, pred, average='micro') # PRAUC
    scores['macro_roc_auc_score'] = roc_auc_score(gold, pred, average='macro')
    scores['micro_roc_auc_score'] = roc_auc_score(gold, pred, average='micro')
    scores['hamming_loss'] = hamming_loss(gold, pred)
    scores['cls_report'] = classification_report(gold, pred)
    return scores



# def store_relevant_cases(relevant_cases, file_path='output/relevant_cases.jsonl'):
#     with open(file_path, 'w') as f:
#         for cases in relevant_cases:
#             line = json.dumps(cases)
#             f.write(line + '\n')
#     return