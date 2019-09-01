# %%
import torch
from torch.utils.data import DataLoader

import csv, argparse

from model import ProjE

TRAIN_DATASET_PATH = './dataset/FB15k/freebase_mtr100_mte100-train.txt'
#TEST_DATASET_PATH = './dataset/FB15k/test.txt'
TEST_DATASET_PATH = './dataset/FB15k/freebase_mtr100_mte100-test.txt'
GPU = torch.cuda.is_available()

def load(file_path):
    with open(file_path, 'r') as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        data = []
        entity_dic = {}
        link_dic = {}
        entity_idx = 0
        link_idx = 0
        for row in tsv_reader: 
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

def load_with_dic(file_path, entity_dic, relation_dic):
    with open(file_path, 'r') as f:
        tsv_reader = csv.reader(f, delimiter='\t')
        data = []
        for row in tsv_reader: 
            head = row[0]
            link = row[1]
            tail = row[2]
            data.append((entity_dic[head], relation_dic[link], entity_dic[tail]))
        return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vector_dim', help='dimension of embedding vector', default=200, type=int)
    parser.add_argument('--sample_p', help='probability of candidate sampling', default=0.5, type=float)
    parser.add_argument('--batch_size', help='batch size for training', default=200, type=int)
    parser.add_argument('--nepoch', help='number of training epoch', default=100, type=int)
    parser.add_argument('--batch_size', help='batch size for training', default=200, type=int)
    parser.add_argument('--lr', help='learning rate', default=0.01, type=float)
    parser.add_argument('--alpha', help='weight for L1 regularlization', default=1e-5, type=float)
    parser.add_argument('--loss_method', help='loss method', default='wlistwise')

    args = parser.parse_args()

    device = torch.device("cuda" if GPU else "cpu")
    train_data, entity_dic, link_dic = load(TRAIN_DATASET_PATH)
    train_data_tensor = torch.tensor(train_data)
    #train_data_tensor = train_data_tensor[:1000]
    test_data = load_with_dic(TEST_DATASET_PATH, entity_dic, link_dic)
    proje = ProjE(device=device,nentity=len(entity_dic),\
        nrelation=len(link_dic), vector_dim=args.vector_dim, sample_p=args.sample_p)
    proje.fit(train_data_tensor, validation=torch.tensor(test_data),\
        batch_size=args.batch_size, nepoch=args.nepoch, lr=args.lr,\
        alpha=args.alpha, loss_method=args.loss_method)
