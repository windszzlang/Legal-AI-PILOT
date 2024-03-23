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
    def __init__(self, max_seq_len, tokenizer, device, is_predict=False):
        self.max_seq_len = max_seq_len
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

        return res


def get_dataloader(data, batch_size, max_seq_len, tokenizer, device, is_shuffle=False, is_predict=False):
    convert_to_features = CustomCollator(max_seq_len, tokenizer, device, is_predict)
    dataset = CustomDataset(data)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_shuffle,
        collate_fn=convert_to_features
    )
    return dataloader