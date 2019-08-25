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
    def __init__(self, nentity, nrelation, device=torch.device('cpu'), vector_dim=200, sample_p=0.5):
        self.nentity = nentity
        self.nrelation = nrelation
        self.vector_dim = vector_dim # k
        self.sample_p = sample_p
        self.sample_n = int(nentity*sample_p)
        self.device = device

    def _candidate_sampling(self, batch):
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
        return Sh, Th, label_h, St, Tt, label_t
    
    #pointwise_loss
    def fit(self, X, batch_size=200, nepoch=1000, lr=0.01, alpha=1e-5, validation=None):
        def init_params(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding) or isinstance(m, nn.Parameter) :
                torch.nn.init.uniform_(m.weight.data, a=-6./(self.vector_dim**(0.5)), b=6./(self.vector_dim**(0.5)))
                #torch.nn.init.uniform(m.bias.data, a=-6./(self.vector_dim**(0.5)), b=6./(self.vector_dim**(0.5)))

        self.X = X
        self.model = ProjENet(nentity=self.nentity, nrelation=self.nrelation, vector_dim=self.vector_dim)
        self.model.apply(init_params)
        self.model = self.model.to(self.device)

        train_loader = torch.utils.data.DataLoader(X, batch_size=batch_size)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for epoch in range(nepoch):
            for batch_idx, batch in enumerate(train_loader):
                if (batch_idx == 1): break
                optimizer.zero_grad()

                Sh, Th, label_h, St, Tt, label_t = self._candidate_sampling(batch)
                Sh = torch.tensor(Sh).to(self.device); Th = torch.stack(Th).to(self.device); label_h = torch.stack(label_h).to(self.device)
                print(Sh[:,0].shape, Sh[:,1].shape, Th.shape)
                h_out_sigmoid_h = self.model(Sh[:,0], Sh[:,1], Th, 0)
                pointwise_loss_h = F.binary_cross_entropy(h_out_sigmoid_h, label_h) 

                St = torch.tensor(St).to(self.device); Tt = torch.stack(Tt).to(self.device); label_t = torch.stack(label_t).to(self.device)
                h_out_sigmoid_t = self.model(St[:,0], St[:,1], Tt, 0)
                pointwise_loss_t = F.binary_cross_entropy(h_out_sigmoid_t, label_t) 

                regu_l1 = 0
                for name, param in self.model.named_parameters():
                    if not name in ['bp', 'bc'] and not 'bias' in name:  
                        regu_l1 += torch.norm(param, 1)
                loss = pointwise_loss_h + pointwise_loss_t + alpha*regu_l1
                print(loss)
                loss.backward()
                optimizer.step()
            if validation is not None:
                print(self.test(validation))

    def predict_relation(self, e1, e2):
        e1_tensor = torch.tensor([e1]).unsqueeze(0)
        e2_tensor = torch.tensor([e2]).unsqueeze(0)
        e1_projection = self.model(e1_tensor)
        e2_projection = self.model(e2_tensor)
        pass

    def predict_entity(self, e, r, candidate, entity_type):
        e_tensor = torch.tensor([e]).unsqueeze(0)
        r_tensor = torch.tensor([r]).unsqueeze(0)
        e_projection = self.model(e_tensor)
        r_projection = self.model(r_tensor)
    
    def save(self, save_path):
        torch.save(self.model.state_dict(), save_path)
    
    def load_model(self, model_path):
        self.model = ProjENet(nentity=self.nentity, nrelation=self.nrelation, vector_dim=self.vector_dim)
        self.model.load_state_dict(torch.load(model_path))
    
    def hits_k(self, x):
        pass

    def test(self, Xtest):
        Xtest = Xtest.to(self.device)
        #hs = Xtest[:,0]; rs = Xtest[:,1]; ts = Xtest[:,2]
        candidate_e = torch.tensor(list(range(self.nentity))).to(self.device)
        #candidate_eb = torch.stack([candidate_e for _ in range(hs.shape[0])])
        candidate_r = torch.tensor(list(range(self.nrelation))).to(self.device)
        #candidate_rb = torch.stack([candidate_r for _ in range(hs.shape[0])])

        hitk_t = 0
        hitk_h = 0
        test_loader = torch.utils.data.DataLoader(Xtest, batch_size=1024)
        for batch_idx, batch in enumerate(test_loader):
            hs = batch[:,0]; rs = batch[:,1]; ts = batch[:,2]
            candidate_eb = torch.stack([candidate_e for _ in range(batch.shape[0])]).to(self.device)
            tail_ranking_score = self.model(hs, rs, candidate_eb, 0)
            rank_idx = torch.argsort(tail_ranking_score, dim=1, descending=True)
            tail_prediction = candidate_e[rank_idx[:,0]]
            hitk_t += torch.sum(tail_prediction==ts)
        #for idx, (h, r, t) in enumerate(Xtest):
        #    h = h.unsqueeze(0); r = r.unsqueeze(0); t = t.unsqueeze(0)
        #    tail_ranking_score = self.model(h, r, candidate_e.unsqueeze(0), 0)
        #    rank_idx = torch.argsort(tail_ranking_score, descending=True)
        #    tail_prediction = candidate_e[rank_idx]
        #    if tail_prediction[0] == t:
        #        hitk_t += 1

        #    head_ranking_score = self.model(t, r, candidate_e.unsqueeze(0), 0)
        #    rank_idx = torch.argsort(head_ranking_score, descending=True)
        #    head_prediction = candidate_e[rank_idx]
        #    if head_prediction[0] == h:
        #        hitk_h += 1

        hitk_t /= Xtest.shape[0]
        hitk_h /= Xtest.shape[0]

        return hitk_t, hitk_h
