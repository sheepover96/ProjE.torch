import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class ProjENet(nn.Module):
    def __init__(self, nentity, nrelation, vector_dim, p_dropout=0.5, *args, **kwargs):
        super(ProjENet, self).__init__(*args, **kwargs)
        self.nentity = nentity #
        self.vector_dim = vector_dim # k
        self.Deh = nn.Linear(self.vector_dim, self.vector_dim, bias=False) 
        self.Drh = nn.Linear(self.vector_dim, self.vector_dim, bias=False) 
        self.Det = nn.Linear(self.vector_dim, self.vector_dim, bias=False) 
        self.Drt = nn.Linear(self.vector_dim, self.vector_dim, bias=False) 
        self.Wr = nn.Embedding(self.nentity, self.vector_dim)
        self.We = nn.Embedding(self.nentity, self.vector_dim)
        self.bc = nn.Parameter(torch.rand(self.vector_dim))
        self.bp = nn.Parameter(torch.rand(1))
        self.sigmoid = nn.Sigmoid() 
        self.tanh = nn.Tanh() 
        self.dropout = nn.Dropout(p=p_dropout)

    #def combination_ope(self, e, r):
    #    return torch.mm(self.De, e) + torch.mm(self.Dr, r) + self.bias


    def forward(self, e, r, samples, entity_type):
        e_emb = self.We(e)
        r_emb = self.Wr(r)
        Wc = self.We(samples)
        if entity_type == 0:
            combination = self.Deh(e_emb) + self.Drh(r_emb) + self.bc
        else:
            combination = self.Det(e_emb) + self.Drt(r_emb) + self.bc
        combination_unsq = combination.unsqueeze(2) 
        h_out_sigmoid = self.sigmoid(torch.bmm(Wc, self.tanh(self.dropout(combination_unsq))) + self.bp)
        h_out_sigmoid_sq = h_out_sigmoid.squeeze()
        return h_out_sigmoid_sq


class ProjE:
    def __init__(self, nentity, nrelation, vector_dim=200, sample_p=0.5):
        self.nentity = nentity
        self.nrelation = nrelation
        self.vector_dim = vector_dim # k
        self.sample_p = sample_p
        self.sample_n = int(nentity*sample_p)

    def _negative_candidate_sampling(self, batch):
        hs = self.X[:,0]; rs = self.X[:,1]; ts = self.X[:,2]
        Sh = []; Th=[]; St = []; Tt = []
        label_h = []
        label_t = []
        for data in batch:
            h = data[0]; r = data[1]; t = data[2]
            e = np.random.choice([h,t])
            if e == h:
                positive_idxs = ((hs == h) * (rs == r)).nonzero().squeeze(1)
                positive_tails = ts[positive_idxs] 
                negative_tails = torch.tensor(np.setdiff1d(ts, positive_tails)) 
                sample_idxs = torch.randperm(len(negative_tails))[:self.sample_n - len(positive_tails)]
                negative_tails = negative_tails[sample_idxs]
                candidate_tail = torch.cat((positive_tails, negative_tails), dim=0)
                Th.append(candidate_tail)
                Sh.append((h, r))
                candidate_label_t = torch.cat((torch.ones(positive_tails.shape[0]), torch.zeros(negative_tails.shape[0])))
                label_h.append(candidate_label_t)
                #negative_samples.append(torch.cat((negative_samples_er, torch.zeros((negative_samples_er.shape[0], 1), dtype=torch.int64)), dim=1))
            else:
                positive_idxs = ((ts == t) * (rs == r)).nonzero().squeeze(1)
                positive_heads = hs[positive_idxs] 
                negative_heads = torch.tensor(np.setdiff1d(hs, positive_heads)) 
                sample_idxs = torch.randperm(len(negative_heads))[:self.sample_n - len(positive_heads)]
                negative_heads = negative_heads[sample_idxs]
                Tt.append(torch.cat((positive_heads, negative_heads), dim=0))
                St.append((h, r))
                candidate_label_h = torch.cat((torch.ones(positive_heads.shape[0]), torch.zeros(negative_heads.shape[0])))
                label_t.append(candidate_label_h)
                #negative_samples.append(torch.cat((negative_samples_er, torch.zeros((negative_samples_er.shape[0], 1), dtype=torch.int64)), dim=1))
        return Sh, Th, label_h, St, Tt, label_t

    #def _negative_candidate_sampling(self, entity, entity_type):
    #    es = self.X[:,0]; rs = self.X[:,1]; ys = self.X[:,2]
    #    negative_idxs = (self.X[:,entity_type]!=entity).nonzero() 
    #    sample_idxs = torch.randperm(len(negative_idxs))[:self.sample_n]
    #    negative_samples = self.X[sample_idxs,:][:,entity_type]
    #    return negative_samples
    
    #def _split_head_and_tail(self, batch):
    #    Sh = []; Th=[]; St = []; Tt = []
    #    for idx, (h, r, t) in enumerate(batch):
    #        e = np.random.choice([h, t])
    #        if e == h:
    #            #Sh.append((e,r))
    #            pos_idxs = (self.X[:,0] == e * self.X[:,1] == r).nonzero()
    #            pos_tail = self.X[pos_idxs][:,2]
    #            neg_tail = self._negative_candidate_sampling(h, 0)
    #        else:
    #            pos_idxs = (self.X[:,2] == e * self.X[:,1] == r).nonzero()
    #            pos_head = self.X[pos_idxs][:,2]
    #            neg_head = self._negative_candidate_sampling(t, 2)
                
    
    def fit(self, X, batch_size=200, nepoch=1000, lr=0.01, alpha=1e-5):
        def init_params(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding) or isinstance(m, nn.Parameter) :
                torch.nn.init.uniform_(m.weight.data, a=-6./(self.vector_dim**(0.5)), b=6./(self.vector_dim**(0.5)))
                #torch.nn.init.uniform(m.bias.data, a=-6./(self.vector_dim**(0.5)), b=6./(self.vector_dim**(0.5)))
        self.X = X
        model = ProjENet(nentity=self.nentity, nrelation=self.nrelation, vector_dim=self.vector_dim)
        model.apply(init_params)
        train_loader = torch.utils.data.DataLoader(X, batch_size=batch_size)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        model.train()
        for epoch in range(nepoch):
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()

                Sh, Th, label_h, St, Tt, label_t = self._negative_candidate_sampling(batch)
                Sh = torch.tensor(Sh); Th = torch.stack(Th); label_h = torch.stack(label_h)
                h_out_sigmoid_h = model(Sh[:,0], Sh[:,1], Th, 0)
                pointwise_loss_h = F.binary_cross_entropy(h_out_sigmoid_h, label_h) 

                St = torch.tensor(St); Tt = torch.stack(Tt); label_t = torch.stack(label_t)
                h_out_sigmoid_t = model(St[:,0], St[:,1], Tt, 0)
                pointwise_loss_t = F.binary_cross_entropy(h_out_sigmoid_t, label_t) 

                regu_l1 = 0
                for name, param in model.named_parameters():
                    if not name in ['bp', 'bc'] and not 'bias' in name:  
                        regu_l1 += torch.norm(param, 1)
                loss = pointwise_loss_h + pointwise_loss_t + alpha*regu_l1
                print(loss)
                loss.backward()
                optimizer.step()
