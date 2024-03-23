import json
import torch
import os
import random
import numpy as np
import json
import re
import evaluate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report



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


def load_data(data_path, tokenizer, load_part=None):
    data = []
    max_len = 0
    with open(data_path) as f:
        for i, line in enumerate(f.readlines()):
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
