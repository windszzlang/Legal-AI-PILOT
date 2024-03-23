import torch
from torch import nn
import torch.nn.functional as F
import math



class Network(nn.Module):
    def __init__(self, fact_encoder, law_encoder, top_k, tokenizer, label_num, label2id, id2label, laws, max_seq_len):
        super().__init__()
        ## basic
        self.fact_encoder = fact_encoder # PLM
        self.law_encoder = law_encoder # law side encoder (PLM)
        # self.top_k = top_k
        self.tokenizer = tokenizer
        self.plm_config = self.fact_encoder.config
        self.label_num = label_num
        self.label2id = label2id
        self.id2label = id2label
        self.max_seq_len = max_seq_len
        self.device = self.fact_encoder.device

        self.last_parameters = None

        self.dropout = nn.Dropout(0.2)

        ## laws side
        self.laws_encodings = self.init_laws_encodings(self.tokenizer, laws)

        ## model structure
        self.law_attention = nn.MultiheadAttention(self.plm_config.hidden_size, num_heads=self.plm_config.num_attention_heads, batch_first=True)
        self.law_attention_W = nn.Linear(self.plm_config.hidden_size, self.plm_config.hidden_size)
        self.tanh = nn.Tanh()

        self.judge_cls =  nn.Linear(self.plm_config.hidden_size + self.label_num, self.label_num)
        # self.judge_cls =  nn.Linear(self.plm_config.hidden_size * 2 + self.label_num, self.label_num)
        self.baseline_cls =  nn.Linear(self.plm_config.hidden_size, self.label_num)

        self.drift_predictor = nn.Sequential(
            nn.Linear(1, self.label_num),
            nn.Tanh(),
            nn.Linear(self.label_num, self.label_num)
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.drift_criterion = nn.MSELoss()
        self.drift_loss_weight = 0.1

        self.baseline = False
        self.use_drift_predictor = True
        self.use_casesifter = True

        self.use_law_encoder = False


    def init_laws_encodings(self, tokenizer, laws):
        encodings = self.tokenizer(
            laws,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            padding='longest',
            return_tensors='pt',
            truncation=True,
            # return_length=True,
        ).to(self.device)
        return {'input_ids': encodings['input_ids'], 'attention_mask': encodings['attention_mask']}
    

    def get_relevant_case_embeddings(self, relevant_cases, fact_embeddings):
        relevant_case_embeddings = []
        for i, r_case in enumerate(relevant_cases):
            case_weights = r_case['scores'].softmax(0).unsqueeze(0) # dim=[1 * k]
            r_labels_embeddings = torch.matmul(case_weights, r_case['labels']) # dim=[1 * label_num]
            r_labels_embeddings = r_labels_embeddings.reshape(-1) # dim=[label_num]
            relevant_case_embeddings.append(r_labels_embeddings) # dim=[label_num]

        return torch.stack(relevant_case_embeddings, dim=0) # dim=[batch_size, hidden_size]


    def forward(self, input_ids, attention_mask, batch_data):
        batch_size = input_ids.size()[0]
        relevant_cases=batch_data['relevant_cases']
        timestamp=batch_data['timestamp']

        last_hidden_state = self.fact_encoder(input_ids, attention_mask, return_dict=True)['last_hidden_state'] # [batch_size, seq_len, hidden_size]
        if self.baseline:
            logits = self.baseline_cls(last_hidden_state[:, 0, :])
            return logits, logits, None

        fact_encoder_output = self.dropout(last_hidden_state) # dim=[batch_size, seq_len, hidden_size]
        fact_embeddings = fact_encoder_output[:, 0, :] # dim=[batch_size, hidden_size]
        # fact_embeddings = self.tanh(fact_embeddings) # cannot activated, need query KB

        if self.use_casesifter:
            relevant_case_embeddings = self.get_relevant_case_embeddings(relevant_cases, fact_embeddings)
        else:
            relevant_case_embeddings = torch.zeros((batch_size, self.label_num)).to(self.device)

        # law encoder
        if self.use_law_encoder:
            last_hidden_state = self.law_encoder(self.laws_encodings['input_ids'], self.laws_encodings['attention_mask'])['last_hidden_state'] # [num_law, seq_len, hidden_size]
            law_embeddings = self.dropout(last_hidden_state[:, 0, :]) # dim=[num_law, hidden_size]

            law_atten_out, law_atten_weights = self.law_attention(fact_embeddings, law_embeddings, law_embeddings) # dim=[batch_size, num_law, hidden_size]
        else:
            law_atten_out = torch.zeros((batch_size, self.plm_config.hidden_size)).to(self.device)
        
        cls_input = torch.cat([fact_embeddings, relevant_case_embeddings], dim=1)
        # cls_input = torch.cat([fact_embeddings, relevant_case_embeddings, law_atten_out], dim=1)
        # cls_input = self.tanh(cls_input)
        logits = self.judge_cls(cls_input)

        drift = self.drift_predictor(timestamp)
        orig_logits = logits
        logits = logits + drift

        return logits, orig_logits, drift


    def compute_loss(self, batch_data):
        self.train()
        logits, orig_logits, drift = self(batch_data['input_ids'], attention_mask=batch_data['attention_mask'], batch_data=batch_data)
        if self.use_drift_predictor:
            loss = self.criterion(logits, batch_data['gold_labels'].float()) # multi-labels cls
            drift_loss = self.drift_criterion(drift, batch_data['gold_labels'].float() - orig_logits.sigmoid())
            loss = (1 - self.drift_loss_weight) * loss + self.drift_loss_weight * drift_loss
        else:
            loss = self.criterion(orig_logits, batch_data['gold_labels'].float()) # multi-labels cls
        return loss


    def predict(self, batch_data):
        self.eval()
        logits, _, _ = self(batch_data['input_ids'], attention_mask=batch_data['attention_mask'], batch_data=batch_data)        

        predictions = torch.zeros_like(logits)
        for i in range(len(predictions)):
            for j in range(len(predictions[0])):
                if logits[i, j].sigmoid() > 0.5:
                    predictions[i, j] = 1

        self.train()
        return predictions