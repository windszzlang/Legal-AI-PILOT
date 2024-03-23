from utils import *
from dataloader import get_dataloader
from network import Network

import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import random
import os


os.environ["CUDA_VISIBLE_DEVICES"] = '4'

# seed = None
seed = 667
save_path = './models/saved/best.pt'
# PLM = 'bert-base-uncased'
# PLM = 'nlpaueb/legal-bert-base-uncased'
# PLM = 'princeton-nlp/unsup-simcse-roberta-large'
# PLM = 'princeton-nlp/unsup-simcse-roberta-base'
PLM = 'princeton-nlp/unsup-simcse-bert-base-uncased'

load_model = True

# default
TOP_K = 5
ALPHA = 6000
# TOP_K = 5 # 3, 5, 7 
# ALPHA = 1e10 # 3000, 6000, 9000, 1e10

if seed != None:
    seed_everything(seed)



def train(epochs=20, patience=3, lr=2e-5, batch_size=16, max_seq_len=512, device='cpu'):

    # prepare labels
    # labels, laws = load_labels('data/ECHR2023/ECHR.json')
    # id2label, label2id, label_num = get_label_id_map(labels)
    # print(id2label)

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(PLM)
    sent_emb = AutoModel.from_pretrained(PLM)
    model = Network(sent_emb, tokenizer, max_seq_len, device, ALPHA).to(device)
    
    # load data and create data lodaer
    train_data = load_data('../data/ECHR2023/train.jsonl', tokenizer)
    valid_data = load_data('../data/ECHR2023/dev.jsonl', tokenizer)
    test_data = load_data('../data/ECHR2023/test.jsonl', tokenizer)
    # pseudo_data = load_data('data/pseudo_data/pseudo_cases.jsonl', tokenizer)
    # train_data = load_data('data/ECHR2023_shuffled/train.jsonl', tokenizer)
    # valid_data = load_data('data/ECHR2023_shuffled/dev.jsonl', tokenizer)
    # test_data = load_data('data/ECHR2023_shuffled/test.jsonl', tokenizer)

    all_data = []
    all_data.extend(train_data)
    all_data.extend(valid_data)
    # all_data.extend(test_data)

    # train_data.extend(valid_data)

    # train_dataloader = get_dataloader(train_data, batch_size, max_seq_len, label2id, label_num, tokenizer, device) #, is_shuffle=True)
    # test_dataloader = get_dataloader(test_data, batch_size, max_seq_len, label2id, label_num, tokenizer, device)
    all_dataloader = get_dataloader(all_data, batch_size, max_seq_len, tokenizer, device)

    # load optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # start training
    if not load_model:
        best_epoch = 0
        best_score = 0.
        patience_cnt = 0
        print('Start Training............')
        for epoch in range(1, epochs + 1):
            patience_cnt += 1
            ## train

            model.train()
            train_loss = 0
            cnt_train = 0
            train_bar = tqdm(all_dataloader, position=0, leave=True)

            for batch_data in train_bar:
                optimizer.zero_grad()
                loss = model.compute_loss(batch_data)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                cnt_train += batch_size
                train_bar.set_description(f'Epoch {epoch}')
                train_bar.set_postfix(loss=train_loss / cnt_train)
        

            print(f'Epoch: [{epoch}/{epochs}], Loss: {train_loss / cnt_train}')
            

        # generate relevant precedent cases
        model.eval()
        torch.save(model, save_path)
    else:
        model = torch.load(save_path)
        model.time_decay_alpha = 1 / ALPHA
        model.eval()
    # model.build_index(train_data)
    # res_data = model.search(test_data, top_k=TOP_K)
    # save_data(res_data, 'data/ECHR2023_gpt/test.jsonl')
    

    # dataset_path = '../data/ECHR2023_ext:' + str(TOP_K) + ':' + str(ALPHA) + '/'
    dataset_path = '../data/ECHR2023_ext/'
    os.mkdir(dataset_path)

    model.build_index(all_data)
    # pseudo_data = model.search(pseudo_data, top_k=TOP_K, start_i=-1)
    # save_data(pseudo_data, 'data/pseudo_data/pseudo_cases_r.jsonl')
    
    
    train_data = model.search(train_data, top_k=TOP_K, start_i=0)
    save_data(train_data, dataset_path + 'train.jsonl')

    valid_data = model.search(valid_data, top_k=TOP_K, start_i=len(train_data))
    save_data(valid_data, dataset_path + 'dev.jsonl')
    
    test_data = model.search(test_data, top_k=TOP_K, start_i=len(train_data) + len(valid_data), test=True)
    save_data(test_data, dataset_path + 'test.jsonl')

    return model



if __name__ == '__main__':
    # model = train(epochs=15, patience=15, lr=2e-5, batch_size=16, max_seq_len=512, device='cuda')
    model = train(epochs=3, patience=1, lr=1e-5, batch_size=16, max_seq_len=512, device='cuda')
