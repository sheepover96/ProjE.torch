# %%
import torch
from torch.utils.data import DataLoader

import csv

from model import ProjE

TRAIN_DATASET_PATH = './dataset/FB15k/freebase_mtr100_mte100-train.txt'
TEST_DATASET_PATH = './dataset/FB15k/test.txt'
GPU = torch.cuda.is_available()

def load(file_path):
    with open(file_path, 'r') as f:
        train_tsv_reader = csv.reader(f, delimiter='\t')
        data = []
        entity_dic = {}
        link_dic = {}
        entity_idx = 0
        link_idx = 0
        for row in train_tsv_reader: 
            head = row[0]
            link = row[1]
            tail = row[2]
            if not head in entity_dic:
                entity_dic[head] = entity_idx
                entity_idx += 1

            if not tail in entity_dic:
                entity_dic[tail] = entity_idx
                entity_idx += 1

            if not link in link_dic:
                link_dic[link] = link_idx
                link_idx += 1
            data.append((entity_dic[head], link_dic[link], entity_dic[tail]))
        return data, entity_dic, link_dic

device = torch.device("cuda" if GPU else "cpu")

train_data, entity_dic, link_dic = load(TRAIN_DATASET_PATH)
test_data, _, _ = load(TEST_DATASET_PATH)

print(len(test_data))
train_data_tensor = torch.tensor(train_data, dtype=torch.int64)
proje = ProjE(device=device,nentity=len(entity_dic), nrelation=len(link_dic))
#proje.fit(train_data_tensor, validation=torch.tensor(test_data))
#print(torch.tensor(train_data))
#train_loader = DataLoader(train_data_tensor, batch_size=64)

#%%


#%%
