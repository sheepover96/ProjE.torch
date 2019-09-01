import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import multiprocessing as multi
from multiprocessing import Manager, Pool, Lock

import time


class ProjENet(nn.Module):
    def __init__(self, nentity, nrelation, vector_dim, p_dropout=0.5, *args, **kwargs):
        super(ProjENet, self).__init__(*args, **kwargs)
        self.nentity = nentity #
        self.vector_dim = vector_dim # k
        #self.Deh = nn.Linear(self.vector_dim, self.vector_dim, bias=False)
        #self.Drh = nn.Linear(self.vector_dim, self.vector_dim, bias=False)
        #self.Det = nn.Linear(self.vector_dim, self.vector_dim, bias=False)
        #self.Drt = nn.Linear(self.vector_dim, self.vector_dim, bias=False)
        self.Deh = nn.Parameter(torch.FloatTensor(self.vector_dim).uniform_(-6/(vector_dim**0.5), 6/(vector_dim**0.5)))
        self.Drh = nn.Parameter(torch.FloatTensor(self.vector_dim).uniform_(-6/(vector_dim**0.5), 6/(vector_dim**0.5)))
        self.Det = nn.Parameter(torch.FloatTensor(self.vector_dim).uniform_(-6/(vector_dim**0.5), 6/(vector_dim**0.5)))
        self.Drt = nn.Parameter(torch.FloatTensor(self.vector_dim).uniform_(-6/(vector_dim**0.5), 6/(vector_dim**0.5)))
        self.Wr = nn.Embedding(self.nentity, self.vector_dim)
        self.We = nn.Embedding(self.nentity, self.vector_dim)
        self.bc = nn.Parameter(torch.rand(self.vector_dim))
        self.bp = nn.Parameter(torch.rand(1))
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(p=p_dropout)

    def forward(self, e, r, samples, entity_type, loss_method='pointwise'):
        e_emb = self.We(e)
        r_emb = self.Wr(r)
        Wc = self.We(samples)
        #if entity_type == 0:
        #    combination = self.Deh(e_emb) + self.Drh(r_emb) + self.bc
        #else:
        #    combination = self.Det(e_emb) + self.Drt(r_emb) + self.bc
        if entity_type == 0:
            combination = self.Deh * e_emb + self.Drh * r_emb + self.bc
        else:
            combination = self.Det * e_emb + self.Drt * r_emb + self.bc
        combination_unsq = combination.unsqueeze(2)
        h_out = torch.bmm(Wc, self.dropout(self.tanh(combination_unsq))) + self.bp
        if loss_method == 'pointwise':
            h_out = self.sigmoid(h_out)
        h_out_sq = h_out.squeeze()
        return h_out_sq


class ProjE:
    def __init__(self, nentity, nrelation, device=torch.device('cpu'), vector_dim=200, sample_p=0.5):
        self.nentity = nentity
        self.nrelation = nrelation
        self.vector_dim = vector_dim # k
        self.sample_p = sample_p
        self.sample_n = int(nentity*sample_p)
        self.device = device

    def _sampling(self, idx, triple, Sh, Th, label_h, St, Tt, label_t, lock):
        h = triple[0]; r = triple[1]; t = triple[2]
        e = np.random.choice([h,t])
        if e == h:
            positive_idxs = ((self.hs == h) * (self.rs == r)).nonzero().squeeze(1)
            positive_tails = self.ts[positive_idxs]
            negative_tails = torch.tensor(np.setdiff1d(self.ts, positive_tails))
            sample_idxs = torch.randperm(len(negative_tails))[:self.sample_n - len(positive_tails)]
            negative_tails = negative_tails[sample_idxs]
            candidate_tail = torch.cat((positive_tails, negative_tails), dim=0)
            candidate_label_t = torch.cat((torch.ones(positive_tails.shape[0]), torch.zeros(negative_tails.shape[0])))
            lock.acquire()
            Th.append(candidate_tail)
            Sh.append((h, r))
            label_h.append(candidate_label_t)
            lock.release()
        else:
            positive_idxs = ((self.ts == t) * (self.rs == r)).nonzero().squeeze(1)
            positive_heads = self.hs[positive_idxs]
            negative_heads = torch.tensor(np.setdiff1d(self.hs, positive_heads))
            sample_idxs = torch.randperm(len(negative_heads))[:self.sample_n - len(positive_heads)]
            negative_heads = negative_heads[sample_idxs]
            candidate_label_h = torch.cat((torch.ones(positive_heads.shape[0]), torch.zeros(negative_heads.shape[0])))
            lock.acquire()
            Tt.append(torch.cat((positive_heads, negative_heads), dim=0))
            St.append((t, r))
            label_t.append(candidate_label_h)
            lock.release()

    def _candidate_sampling_mp_batch(self, batch):
        manager = Manager()
        Sh = manager.list(); Th=manager.list(); label_h = manager.list()
        St = manager.list(); Tt=manager.list(); label_t = manager.list()
        pool = Pool(multi.cpu_count())
        lock = Lock()
        for idx, triple in enumerate(batch):
            pool.apply_async(self._sampling, (idx, triple, Sh, Th, label_h, St, Tt, label_t, lock))
        pool.close()
        pool.join()
        return Sh, Th, label_h, St, Tt, label_t

    def _candidate_sampling_mp(self):
        manager = Manager()
        Sh = manager.list(); Th=manager.list(); label_h = manager.list()
        St = manager.list(); Tt=manager.list(); label_t = manager.list()
        pool = Pool(4)
        for idx, triple in enumerate(self.X):
            pool.apply_async(self._sampling, (idx, triple, Sh, Th, label_h, St, Tt, label_t))
        pool.close()
        pool.join()
        return Sh, Th, label_h, St, Tt, label_t

    def _cache_sample_candidate(self):
        self.hr_dic_pos = {}; self.tr_dic_pos = {}
        self.hr_dic_neg = {}; self.tr_dic_neg = {}
        for idx, (h, r, t) in enumerate(self.X):
            h = h.item(); r = r.item(); t = t.item()
            #print(h,r,t)
            if not (h, r) in self.hr_dic_pos:
                self.hr_dic_pos[(h, r)] = [t]
            else:
                self.hr_dic_pos[(h, r)].append(t)

            if not (t, r) in self.tr_dic_pos:
                self.tr_dic_pos[(t, r)] = [h]
            else:
                self.tr_dic_pos[(t, r)].append(h)
        entities = torch.arange(self.nentity)
        for key in self.hr_dic_pos.keys():
            positive_tails = self.hr_dic_pos[key]
            self.hr_dic_neg[key] = torch.tensor(np.setdiff1d(entities, positive_tails))
            self.hr_dic_pos[key] = torch.tensor(positive_tails)

        for key in self.tr_dic_pos.keys():
            positive_heads = self.tr_dic_pos[key]
            self.tr_dic_neg[key] = torch.tensor(np.setdiff1d(entities, positive_heads))
            self.tr_dic_pos[key] = torch.tensor(positive_heads)

    def _candidate_sampling_with_cache(self, batch):
        hs = self.X[:,0]; rs = self.X[:,1]; ts = self.X[:,2]
        Sh = []; Th=[]; St = []; Tt = []
        label_h = []
        label_t = []
        for (h, r, t) in batch:
            h = h.item(); r = r.item(); t = t.item()
            e = np.random.choice([h,t])
            if e == h:
                positive_tails = self.hr_dic_pos[(h, r)]
                negative_tails = self.hr_dic_neg[(h, r)]
                sample_idxs = torch.randperm(len(negative_tails))[:self.sample_n - len(positive_tails)]
                negative_tails = negative_tails[sample_idxs]
                candidate_tail = torch.cat((positive_tails, negative_tails), dim=0)
                Th.append(candidate_tail)
                Sh.append((h, r))
                candidate_label_t = torch.cat((torch.ones(positive_tails.shape[0]), torch.zeros(negative_tails.shape[0])))
                label_h.append(candidate_label_t)
            else:
                positive_heads = self.tr_dic_pos[(t, r)]
                negative_heads = self.tr_dic_neg[(t, r)]
                sample_idxs = torch.randperm(len(negative_heads))[:self.sample_n - len(positive_heads)]
                negative_heads = negative_heads[sample_idxs]
                Tt.append(torch.cat((positive_heads, negative_heads), dim=0))
                St.append((t, r))
                candidate_label_h = torch.cat((torch.ones(positive_heads.shape[0]), torch.zeros(negative_heads.shape[0])))
                label_t.append(candidate_label_h)
        return Sh, Th, label_h, St, Tt, label_t

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
                St.append((t, r))
                candidate_label_h = torch.cat((torch.ones(positive_heads.shape[0]), torch.zeros(negative_heads.shape[0])))
                label_t.append(candidate_label_h)
        return Sh, Th, label_h, St, Tt, label_t

    def _sampled_softmax(self, tensor, weights):
        max_val = torch.max(tensor, dim=1)[0]
        tensor_rescaled = torch.t(torch.t(tensor) - max_val)
        tensor_exp = torch.exp(tensor_rescaled)
        tensor_sum = torch.sum(tensor_exp * torch.abs(weights), dim=1)
        return (tensor_exp/tensor_sum) * torch.abs(weights)

    def fit(self, X, batch_size=200, nepoch=100, lr=0.01, alpha=1e-5, validation=None, loss_method='wlistwise'):
        def init_params(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding) or isinstance(m, nn.Parameter) :
                torch.nn.init.uniform_(m.weight.data, a=-6./(self.vector_dim**(0.5)), b=6./(self.vector_dim**(0.5)))
                #torch.nn.init.uniform(m.bias.data, a=-6./(self.vector_dim**(0.5)), b=6./(self.vector_dim**(0.5)))

        self.X = X
        self.hs = self.X[:,0]; self.rs = self.X[:,1]; self.ts = self.X[:,2]
        self._cache_sample_candidate()
        self.model = ProjENet(nentity=self.nentity, nrelation=self.nrelation, vector_dim=self.vector_dim)
        self.model.apply(init_params)
        self.model = self.model.to(self.device)
        train_loader = torch.utils.data.DataLoader(X, batch_size=batch_size)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for epoch in range(nepoch):
            batch_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()

                Sh, Th, label_h, St, Tt, label_t = self._candidate_sampling_with_cache(batch)
                Sh = torch.tensor(Sh).to(self.device); Th = torch.stack(Th).to(self.device); label_h = torch.stack(label_h).to(self.device)
                h_out_h = self.model(Sh[:,0], Sh[:,1], Th, 0)

                St = torch.tensor(St).to(self.device); Tt = torch.stack(Tt).to(self.device); label_t = torch.stack(label_t).to(self.device)
                h_out_t = self.model(St[:,0], St[:,1], Tt, 2)

                if loss_method == 'pointwise':
                    loss_h = F.binary_cross_entropy(h_out_h, label_h, reduction='sum')
                    loss_t = F.binary_cross_entropy(h_out_t, label_t, reduction='sum')
                elif loss_method == 'listwise':
                    h_out_h_sm = self._sampled_softmax(h_out_h, label_h)
                    loss_h = -torch.sum(torch.log(torch.clip(h_out_h_sm, 1e-10, 1.)) * label_h)
                    h_out_t_sm = self._sampled_softmax(h_out_t, label_t)
                    loss_t = -torch.sum(torch.log(torch.clip(h_out_t_sm, 1e-10, 1.)) * label_t)
                else: #wlistwise
                    h_out_h_sm = self._sampled_softmax(h_out_h, label_h)
                    loss_h = -torch.sum(torch.log(torch.clip(h_out_h_sm, 1e-10, 1.)) * label_h / torch.sum(label_h, dim=1))
                    h_out_t_sm = self._sampled_softmax(h_out_t, label_t)
                    loss_t = -torch.sum(torch.log(torch.clip(h_out_t_sm, 1e-10, 1.)) * label_t / torch.sum(label_t, dim=1))

                regu_l1 = 0
                for name, param in self.model.named_parameters():
                    if not name in ['bp', 'bc'] and not 'bias' in name:
                        regu_l1 += torch.norm(param, 1)
                loss = loss_h + loss_t + alpha*regu_l1
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
            print('epoch', epoch+1, batch_loss/len(train_loader))
            print(self.test(X[:10000]))
            if validation is not None and (epoch + 1)%1 == 0:
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
        self.model.eval()
        Xtest = Xtest.to(self.device)
        #hs = Xtest[:,0]; rs = Xtest[:,1]; ts = Xtest[:,2]
        candidate_e = torch.tensor(list(range(self.nentity))).to(self.device)
        #candidate_eb = torch.stack([candidate_e for _ in range(hs.shape[0])])
        candidate_r = torch.tensor(list(range(self.nrelation))).to(self.device)
        #candidate_rb = torch.stack([candidate_r for _ in range(hs.shape[0])])

        hitk_t = 0; mean_rank_t = 0
        hitk_h = 0; mean_rank_h = 0
        test_loader = torch.utils.data.DataLoader(Xtest, batch_size=128)
        for batch_idx, batch in enumerate(test_loader):
            hs = batch[:,0]; rs = batch[:,1]; ts = batch[:,2]
            candidate_eb = torch.stack([candidate_e for _ in range(batch.shape[0])]).to(self.device)
            tail_ranking_score = self.model(hs, rs, candidate_eb, 0)
            rank_idxs = torch.argsort(tail_ranking_score, dim=1, descending=True)
            for idx, rank_idx in enumerate(rank_idxs):
                tail_prediction = candidate_e[rank_idx]
                t = ts[idx]
                rank = ((tail_prediction == t).nonzero()).squeeze().item()
                mean_rank_t += rank
                if t in tail_prediction[:10]:
                    hitk_t += 1

        for batch_idx, batch in enumerate(test_loader):
            hs = batch[:,0]; rs = batch[:,1]; ts = batch[:,2]
            candidate_eb = torch.stack([candidate_e for _ in range(batch.shape[0])]).to(self.device)
            head_ranking_score = self.model(ts, rs, candidate_eb, 2)
            rank_idxs = torch.argsort(head_ranking_score, dim=1, descending=True)
            for idx, rank_idx in enumerate(rank_idxs):
                head_prediction = candidate_e[rank_idx]
                h = hs[idx]
                rank = ((head_prediction == h).nonzero()).squeeze().item()
                mean_rank_h += rank
                if h in head_prediction[:10]:
                    hitk_h += 1

        hitk_t = (hitk_t / Xtest.shape[0]) * 100
        hitk_h = (hitk_h / Xtest.shape[0]) * 100
        mean_rank_t /= Xtest.shape[0]
        mean_rank_h /= Xtest.shape[0]

        return hitk_t, mean_rank_t, hitk_h, mean_rank_h
