import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm



def simcse_loss(pred, gold=None):
    idxs = torch.arange(0, pred.size()[0])
    idxs_1 = idxs.unsqueeze(0)
    # positive sample is the prior one if index is even, else the last
    idxs_2 = (idxs + 1 - idxs % 2 * 2).unsqueeze(1)
    gold = torch.isclose(idxs_1, idxs_2).float().type_as(pred)
    # compute similarities
    pred = F.normalize(pred, dim=1, p=2)
    # similarities = cosine_similarity(pred, pred)
    similarities = torch.matmul(pred, pred.T)
    similarities = similarities - torch.eye(pred.size()[0]).type_as(pred) * 1e12
    similarities = similarities * 20
    loss = F.cross_entropy(similarities, gold, reduction='mean')
    return loss


class Network(nn.Module):
    def __init__(self, sent_emb, tokenizer, max_seq_len, device, alpha=1e10):
        super().__init__()
        self.sent_emb = sent_emb
        self.bert_config = self.sent_emb.config

        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.device = device
        
        self.case_list = []
        self.case_embeddings = []

        # self.criterion = nn.CrossEntropyLoss()
        self.cosine_similarity_dim0 = nn.CosineSimilarity(dim=0) # scope=[-1, 1]
        self.cosine_similarity_dim1 = nn.CosineSimilarity(dim=1) # scope=[-1, 1]

        self.time_decay_alpha = 1 / alpha


    def forward(self, input_ids, attention_mask):
        embeddings = self.sent_emb(input_ids, attention_mask=attention_mask).pooler_output # shape=[batch_size, seq_len, vocab_size]
        return embeddings


    def compute_loss(self, batch_data):
        self.train()
        batch_size = batch_data['input_ids'].size()[0]
        input_ids, attention_mask = [], []
        for i in range(batch_size): # create copies
            input_ids.append(batch_data['input_ids'][i])
            input_ids.append(batch_data['input_ids'][i])
            attention_mask.append(batch_data['attention_mask'][i])
            attention_mask.append(batch_data['attention_mask'][i])
        input_ids = torch.stack(input_ids, dim=0)
        attention_mask = torch.stack(attention_mask, dim=0)
        # compute pred / similarity
        pred = self(input_ids, attention_mask)
        loss = simcse_loss(pred)

        return loss


    def build_index(self, case_list):
        self.eval()
        self.case_list = case_list
        
        print('Building index......')
        for tmp_case in tqdm(self.case_list):
            facts = tmp_case['facts']
            text_encodings = self.tokenizer(facts,
                add_special_tokens=True,
                max_length=self.max_seq_len,
                padding='longest',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = text_encodings['input_ids'].to(self.device)
            attention_mask = text_encodings['attention_mask'].to(self.device)
            with torch.no_grad():
                embeddings = self(input_ids, attention_mask)[0]
            self.case_embeddings.append(embeddings)
        self.case_embeddings = torch.stack(self.case_embeddings, dim=0)
        self.train()
        return


    def search(self, case_list, top_k, start_i=0, test=False):
        print('Searching......')
        self.eval()
        for i, tmp_case in enumerate(tqdm(case_list)):
            facts = tmp_case['facts']
            text_encodings = self.tokenizer(facts,
                add_special_tokens=True,
                max_length=self.max_seq_len,
                padding='longest',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = text_encodings['input_ids'].to(self.device)
            attention_mask = text_encodings['attention_mask'].to(self.device)
            with torch.no_grad():
                embeddings = self(input_ids, attention_mask)[0]
            sims = self.cosine_similarity_dim1(embeddings, self.case_embeddings)
            if start_i != -1: # when start_i = -1, find relevant cases in all cases
                for j in range(start_i + i, len(sims)):
                    if j < 100 and j != start_i + i:
                        continue
                    sims[j] = -1.

            # time decay
            if test:
                for j in range(0, start_i):
                    sims[j] *= 1 / (1 + self.time_decay_alpha * (start_i + i - 1 - j))
            else:
                for j in range(0, start_i + i):
                    sims[j] *= 1 / (1 + self.time_decay_alpha * (start_i + i - 1 - j))

            values, indices = torch.topk(sims, top_k)
            indices = indices.cpu().tolist()
            case_list[i]['relevant_cases'] = {
                'case_ids': [self.case_list[idx]['case_id'] for idx in indices],
                'scores': values.cpu().tolist(),
                'facts': [self.case_list[idx]['facts'] for idx in indices],
                'violated_articles': [self.case_list[idx]['violated_articles'] for idx in indices],
            }
        self.train()
        return case_list



#    def compute_loss(self, batch_data):
#         self.train()
#         batch_size = batch_data['input_ids'].size()[0]
        
#         # compute pred / similarity
#         doc_emb = self(batch_data['input_ids'], batch_data['attention_mask'])
#         doc_emb = F.normalize(doc_emb, dim=1, p=2)

#         sims = torch.zeros((batch_size, batch_size))
#         for i in range(batch_size):
#             for j in range(batch_size):
#                 sims[i, j] = self.cosine_similarity_dim0(doc_emb[i], doc_emb[j])
#         pred = F.normalize(sims, dim=1, p=2) # x/norm(x): unit vector, remove influence of different docs between batch
#         pred = pred - torch.eye(sims.size()[0]).type_as(pred) * 1e12 # do not consider itself
#         pred /= 0.4 # temperature: less T means more attention to hard negative samples (similar to positive samples), scope=[0, 1]
        
#         # compute gold
#         gold = torch.zeros((batch_size, batch_size))
#         for i in range(batch_size):
#             for j in range(batch_size):
#                 if i == j:
#                     gold[i, j] = 1.
#         loss = self.criterion(pred, gold)
#         return loss