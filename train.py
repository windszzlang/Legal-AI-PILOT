from utils import *
from dataloader import get_dataloader
from network import *

import torch
from transformers import AutoTokenizer, AutoModel
# from transformers import get_cosine_schedule_with_warmup
import pytorch_warmup
from tqdm import tqdm
import random



os.environ["CUDA_VISIBLE_DEVICES"] = '4'

# seed = None
seed_pool = [667, 123, 1024, 199, 1]
seed = seed_pool[4]
# para exp seed lambda
# seed = 999
# para exp2 seed casesifer
# seed = 444
save_path = './models/saved/best.pt'
# PLM = 'bert-base-uncased'
PLM = 'nlpaueb/legal-bert-base-uncased'
dataset = 'ECHR2023_ext'

# dataset = 'ECHR2023_ext:7:' + str(1e10)
# dataset = 'ECHR2023_ext:7:6000'

# nonsense
TOP_K = None

if seed != None:
    seed_everything(seed)



def train(epochs=20, patience=3, lr=2e-5, batch_size=16, max_seq_len=512, device='cpu'):

    # prepare labels
    labels, laws = load_labels('data/ECHR2023/ECHR.json')
    id2label, label2id, label_num = get_label_id_map(labels)
    print(id2label)
    print('Seed: ' + str(seed))

    # load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(PLM)
    fact_encoder = AutoModel.from_pretrained(PLM).to(device)
    law_encoder = AutoModel.from_pretrained(PLM).to(device)
    model = Network(fact_encoder, law_encoder, TOP_K, tokenizer, label_num, label2id, id2label, laws, max_seq_len).to(device)
    
    # load data and create data lodaer
    train_data = load_data('data/' + dataset + '/train.jsonl', tokenizer)
    valid_data = load_data('data/' + dataset + '/dev.jsonl', tokenizer)
    test_data = load_data('data/' + dataset + '/test.jsonl', tokenizer)
    data_num = len(train_data) + len(valid_data) + len(test_data)
    train_dataloader = get_dataloader(train_data, batch_size, max_seq_len, label2id, label_num, data_num, tokenizer, device, is_shuffle=True)
    valid_dataloader = get_dataloader(valid_data, batch_size, max_seq_len, label2id, label_num, data_num, tokenizer, device)
    test_dataloader = get_dataloader(test_data, batch_size, max_seq_len, label2id, label_num, data_num, tokenizer, device)

    # load optimizer
    epsilon = 10
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer = torch.optim.AdamW([
        {'params': model.fact_encoder.parameters(), 'lr': lr},
        {'params': model.law_encoder.parameters(), 'lr': lr},
        {'params': model.law_attention.parameters(), 'lr': lr},
        {'params': model.judge_cls.parameters(), 'lr': lr * epsilon * epsilon},
        {'params': model.drift_predictor.parameters(), 'lr': lr},
        {'params': model.baseline_cls.parameters(), 'lr': lr}
    ], lr=lr)
    # lr schedule
    max_steps = len(train_dataloader) * 10
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps) # T_max: Maximum number of iterations.
    warmup_scheduler = pytorch_warmup.RAdamWarmup(optimizer)

    # start training
    best_epoch = 0
    best_score = 0.
    patience_cnt = 0
    print('Start Training............')
    for epoch in range(1, epochs + 1):
        patience_cnt += 1

        model.train()
        train_loss = 0
        cnt_train = 0
        train_bar = tqdm(train_dataloader, position=0, leave=True)

        for batch_data in train_bar:
            optimizer.zero_grad()
            loss = model.compute_loss(batch_data)
            loss.backward()
            optimizer.step()

            with warmup_scheduler.dampening():
                lr_scheduler.step()


            train_loss += loss.item() # + loss_2.item()
            cnt_train += batch_size
            train_bar.set_description(f'Epoch {epoch}')
            train_bar.set_postfix(loss=train_loss / cnt_train)
    
        print(f'Epoch: [{epoch}/{epochs}], Loss: {train_loss / cnt_train}')
        ## valid
        model.eval()
        pred = []
        gold = []
        with torch.no_grad():
            valid_bar = tqdm(valid_dataloader, position=0, leave=True)
            for batch_data in valid_bar:
                predictions = model.predict(batch_data)
                pred.extend(predictions)
                gold.extend(batch_data['gold_labels'])
                valid_bar.set_description(f'Epoch {epoch}')
        
        article_scores = compute_article_performance(pred, gold)
        print(f'Epoch [{epoch}/{epochs}] dev:')
        print(f'macro_f1_score {article_scores["macro_f1_score"]}, micro_f1_score {article_scores["micro_f1_score"]}, precision {article_scores["precision_score"]}, recall {article_scores["recall_score"]}')
        print(f'macro_jaccard_score {article_scores["macro_jaccard_score"]}, micro_jaccard_score {article_scores["micro_jaccard_score"]}')
        print(f'macro_PRAUC {article_scores["macro_PRAUC"]}, micro_PRAUC {article_scores["micro_PRAUC"]}, hamming_loss {article_scores["hamming_loss"]}')
        print(f'macro_roc_auc_score: {article_scores["macro_roc_auc_score"]}, micro_roc_auc_score: {article_scores["micro_roc_auc_score"]}')
        print(article_scores['cls_report'])
        cur_score = article_scores['micro_f1_score']

        ## test
        pred = []
        gold = []
        with torch.no_grad():
            test_bar = tqdm(test_dataloader, position=0, leave=True)
            for batch_data in test_bar:
                predictions = model.predict(batch_data)
                pred.extend(predictions)
                gold.extend(batch_data['gold_labels'])
                test_bar.set_description(f'Epoch {epoch}')
        
        article_scores = compute_article_performance(pred, gold)
        print(f'Epoch [{epoch}/{epochs}] test:')
        print(f'macro_f1_score {article_scores["macro_f1_score"]}, micro_f1_score {article_scores["micro_f1_score"]}, precision {article_scores["precision_score"]}, recall {article_scores["recall_score"]}')
        print(f'macro_jaccard_score {article_scores["macro_jaccard_score"]}, micro_jaccard_score {article_scores["micro_jaccard_score"]}')
        print(f'macro_PRAUC {article_scores["macro_PRAUC"]}, micro_PRAUC {article_scores["micro_PRAUC"]}, hamming_loss {article_scores["hamming_loss"]}')
        print(f'macro_roc_auc_score: {article_scores["macro_roc_auc_score"]}, micro_roc_auc_score: {article_scores["micro_roc_auc_score"]}')
        print(article_scores['cls_report'])
        # cur_score = article_scores['f1_score']


        if cur_score > best_score:
            patience_cnt = 0
            best_score = cur_score
            best_epoch = epoch
            # checkpoint = {'net': model.state_dict(), 'optimizer':optimizer.state_dict(), 'epoch': epoch}
            # torch.save(checkpoint, save_path)
            torch.save(model, save_path)
            print('***** new score *****')
            print(f'The best epoch is: {best_epoch}, with the best score is: {best_score}')
            print('********************')
        elif patience_cnt >= patience: # early stop
            # if stage == 'init':
            #     stage = 'train'
            #     patience_cnt = 0
            # elif stage == 'train':
            print(f'Early Stop with best epoch {best_epoch}.......')
            break

    # test
    test_model = torch.load(save_path)
    test_model.eval()
    pred = []
    gold = []
    with torch.no_grad():
        test_bar = tqdm(test_dataloader, position=0, leave=True)
        for batch_data in test_bar:
            predictions = test_model.predict(batch_data)
            pred.extend(predictions)
            gold.extend(batch_data['gold_labels'])
        
    article_scores = compute_article_performance(pred, gold)
    print(f'Test:')
    print(f'macro_f1_score {article_scores["macro_f1_score"]}, micro_f1_score {article_scores["micro_f1_score"]}, precision {article_scores["precision_score"]}, recall {article_scores["recall_score"]}')
    print(f'macro_jaccard_score {article_scores["macro_jaccard_score"]}, micro_jaccard_score {article_scores["micro_jaccard_score"]}')
    print(f'macro_PRAUC {article_scores["macro_PRAUC"]}, micro_PRAUC {article_scores["micro_PRAUC"]}, hamming_loss {article_scores["hamming_loss"]}')
    print(f'macro_roc_auc_score: {article_scores["macro_roc_auc_score"]}, micro_roc_auc_score: {article_scores["micro_roc_auc_score"]}')
    print(article_scores['cls_report'])

    return test_model



if __name__ == '__main__':
    # model = train(epochs=15, patience=15, lr=2e-5, batch_size=8, max_seq_len=512, device='cuda')
    model = train(epochs=10, patience=10, lr=1e-5, batch_size=8, max_seq_len=512, device='cuda')
    # model = train(epochs=10, patience=2, lr=1e-5, batch_size=8, max_seq_len=512, device='cuda')
