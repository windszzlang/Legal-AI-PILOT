import torch
from torch.utils.data import Dataset, DataLoader



class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        assert idx < len(self.data)
        return self.data[idx]

    def __len__(self):
        return len(self.data)
    

class CustomCollator():
    def __init__(self, max_seq_len, label2id, label_num, data_num, tokenizer, device, is_predict=False):
        self.max_seq_len = max_seq_len
        self.label2id = label2id
        self.label_num = label_num
        self.data_num = data_num
        self.tokenizer = tokenizer
        self.device = device
        self.is_predict = is_predict

    def __call__(self, batch_data):

        # batch size must be 1
        batch_text = [D['facts'] for D in batch_data]

        text_encodings = self.tokenizer(
            batch_text,
            add_special_tokens=True,
            max_length=self.max_seq_len,
            padding='longest',
            return_tensors='pt',
            truncation=True,
            # return_length=True,
        )
        res = {}
        res['input_ids'] = text_encodings['input_ids'].to(self.device)
        res['attention_mask'] = text_encodings['attention_mask'].to(self.device)
        
        
        # res['timestamp'] = torch.tensor([[D['time_id'] / self.data_num] for D in batch_data]).to(self.device)
        res['timestamp'] = torch.tensor([[D['time_id'] * 1.0] for D in batch_data]).to(self.device)
        # res['timestamp'] = torch.tensor([[float(D['judgement_date'].split(' ')[0][-4:])] for D in batch_data]).to(self.device)
        res['relevant_cases'] = []
        for D in batch_data:
            relevant_cases = D['relevant_cases']
            text_encodings = self.tokenizer(
                relevant_cases['facts'],
                add_special_tokens=True,
                max_length=self.max_seq_len,
                padding='longest',
                return_tensors='pt',
                truncation=True,
                # return_length=True,
            )
            labels = torch.zeros(len(relevant_cases['case_ids']), self.label_num).to(self.device)
            for i, vs in enumerate(relevant_cases['violated_articles']):
                for v in vs:
                    labels[i, self.label2id[v]] = 1
            res['relevant_cases'].append({
                'input_ids': text_encodings['input_ids'].to(self.device),
                'attention_mask': text_encodings['attention_mask'].to(self.device),
                'labels': labels,
                'scores': torch.tensor(D['relevant_cases']['scores']).to(self.device)
            })

        if not self.is_predict:
            res['gold_labels'] = torch.zeros(len(batch_data), self.label_num).to(self.device)
            for i, D in enumerate(batch_data):
                for article in D['violated_articles']:
                    res['gold_labels'][i, self.label2id[article]] = 1
        return res


def get_dataloader(data, batch_size, max_seq_len, label2id, label_num, data_num, tokenizer, device, is_shuffle=False, is_predict=False):
    convert_to_features = CustomCollator(max_seq_len, label2id, label_num, data_num, tokenizer, device, is_predict)
    dataset = CustomDataset(data)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_shuffle,
        collate_fn=convert_to_features
    )
    return dataloader