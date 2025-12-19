

import torch
import torch.sparse as torch_sp
import torch.nn as nn
import torch.nn.functional as F
from model.base import AbstractRecommender
from util.pytorch import inner_product
from util.pytorch import get_initializer
from data import PairwiseSampler
import numpy as np
import time
from reckit import timer
import scipy.sparse as sp
from util.common import normalize_adj_matrix
from util.pytorch import sp_mat_to_sp_tensor


class _XSimGCL(nn.Module):
    def __init__(self, num_users, num_items, embed_dim, norm_adj, n_layers, layer_cl, eps, device):
        super(_XSimGCL, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.norm_adj = norm_adj
        self.n_layers = n_layers
        self.layer_cl = layer_cl
        self.eps = eps
        self.device = device
        self.user_embeddings = nn.Embedding(num_users, embed_dim)
        self.item_embeddings = nn.Embedding(num_items, embed_dim)
        self._user_embeddings_final = None
        self._item_embeddings_final = None

        # weight initialization
        self.reset_parameters()

    def reset_parameters(self, init_method="uniform"):
        init = get_initializer(init_method)
        init(self.user_embeddings.weight)
        init(self.item_embeddings.weight)

    def forward(self, users, items):
        user_embeddings, item_embeddings, _, _ = self._forward_gcn()
        user_embs = F.embedding(users, user_embeddings)
        item_embs = F.embedding(items, item_embeddings)
        ratings = inner_product(user_embs, item_embs)
        return ratings

    def _forward_gcn(self):
        ego_embeddings = torch.cat([self.user_embeddings.weight, self.item_embeddings.weight], dim=0)
        all_embeddings = []
        all_embeddings_cl = ego_embeddings
        for k in range(self.n_layers):
            ego_embeddings = torch_sp.mm(self.norm_adj, ego_embeddings)
            random_noise = torch.rand_like(ego_embeddings).cuda()
            ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings.append(ego_embeddings)
            if k == self.layer_cl - 1:
                all_embeddings_cl = ego_embeddings
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(final_embeddings,
                                                               [self.num_users, self.num_items])
        user_all_embeddings_cl, item_all_embeddings_cl = torch.split(all_embeddings_cl,
                                                                     [self.num_users, self.num_items])

        return user_all_embeddings, item_all_embeddings, user_all_embeddings_cl, item_all_embeddings_cl

    def predict(self, users):
        if self._user_embeddings_final is None or self._item_embeddings_final is None:
            raise ValueError("Please first switch to 'eval' mode.")
        user_embs = F.embedding(users, self._user_embeddings_final)
        ratings = torch.matmul(user_embs, self._item_embeddings_final.T)
        return ratings

    def eval(self):
        super(_XSimGCL, self).eval()
        self._user_embeddings_final, self._item_embeddings_final, _, _ = self._forward_gcn()




class XSimGCL(AbstractRecommender):
    def __init__(self, config):
        super(XSimGCL, self).__init__(config)
        self.lr = config['lr']
        self.reg = config['reg']
        self.emb_size = config['embed_size']
        self.batch_size = config['batch_size']
        self.epochs = config["epochs"]
        self.n_layers = config['n_layers']
        self.adj_type = config["adj_type"]
        self.cl_rate = config["lambda"]
        self.eps = config["eps"]
        self.temp = config["temp"]
        self.param_init = config["param_init"]
        self.is_pairwise = config["is_pairwise"]
        self.loss_func = config["loss_func"]
        self.layer_cl = 1

        self.num_users, self.num_items = self.dataset.num_users, self.dataset.num_items

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        adj_matrix = self.create_adj_mat(self.adj_type)
        adj_matrix = sp_mat_to_sp_tensor(adj_matrix).to(self.device)

        self.XSimGCL = _XSimGCL(self.num_users, self.num_items, self.emb_size,
                                  adj_matrix, self.n_layers, self.layer_cl, self.eps, self.device).to(self.device)
        self.XSimGCL.reset_parameters(self.param_init)
        self.optimizer = torch.optim.Adam(self.XSimGCL.parameters(), lr=self.lr)

    def InfoNCE(self, view1, view2, temperature: float, b_cos: bool = True):
        """
        Args:
            view1: (torch.Tensor - N x D)
            view2: (torch.Tensor - N x D)
            temperature: float
            b_cos (bool)

        Return: Average InfoNCE Loss
        """
        if b_cos:
            view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)

        pos_score = (view1 @ view2.T) / temperature
        score = torch.diag(F.log_softmax(pos_score, dim=1))
        return -score.mean()

    def bpr_loss(self, user_emb, pos_item_emb, neg_item_emb):

        pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
        neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
        loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
        return torch.mean(loss)

    def l2_reg_loss(self, reg, *args):
        emb_loss = 0
        for emb in args:
            emb_loss += torch.norm(emb, p=2) / emb.shape[0]
        return emb_loss * reg

    def cal_cl_loss(self, idx, user_view1, user_view2, item_view1, item_view2):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).cuda()
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).cuda()

        user_cl_loss = self.InfoNCE(user_view1[u_idx], user_view2[u_idx], self.temp)
        item_cl_loss = self.InfoNCE(item_view1[i_idx], item_view2[i_idx], self.temp)
        return user_cl_loss + item_cl_loss

    @timer
    def create_adj_mat(self, adj_type):
        users_items = self.dataset.train_data.to_user_item_pairs()
        users_np, items_np = users_items[:, 0], users_items[:, 1]

        ratings = np.ones_like(users_np, dtype=np.float32)
        n_nodes = self.num_users + self.num_items
        up_left_adj = sp.csr_matrix((ratings, (users_np, items_np+self.num_users)), shape=(n_nodes, n_nodes))
        adj_mat = up_left_adj + up_left_adj.T

        if adj_type == 'plain':
            adj_matrix = adj_mat
            print('use the plain adjacency matrix')
        elif adj_type == 'norm':
            adj_matrix = normalize_adj_matrix(adj_mat + sp.eye(adj_mat.shape[0]), norm_method="left")
            print('use the normalized adjacency matrix')
        elif adj_type == 'gcmc':
            adj_matrix = normalize_adj_matrix(adj_mat, norm_method="left")
            print('use the gcmc adjacency matrix')
        elif adj_type == 'pre':
            # pre adjcency matrix
            adj_matrix = normalize_adj_matrix(adj_mat, norm_method="symmetric")
            print('use the pre adjcency matrix')
        else:
            mean_adj = normalize_adj_matrix(adj_mat, norm_method="left")
            adj_matrix = mean_adj + sp.eye(mean_adj.shape[0])
            print('use the mean adjacency matrix')

        return adj_matrix

    def train_model(self):
        data_iter = PairwiseSampler(self.dataset.train_data, num_neg=1,
                                    batch_size=self.batch_size,
                                    shuffle=True, drop_last=False)
        self.logger.info(self.evaluator.metrics_info())
        training_start_time = time.time()
        for epoch in range(self.epochs):
            self.XSimGCL.train()
            for bat_users, bat_pos_items, bat_neg_items in data_iter:
                bat_users = torch.from_numpy(bat_users).long().to(self.device)
                bat_pos_items = torch.from_numpy(bat_pos_items).long().to(self.device)
                bat_neg_items = torch.from_numpy(bat_neg_items).long().to(self.device)
                rec_user_emb, rec_item_emb, cl_user_emb, cl_item_emb = self.XSimGCL._forward_gcn()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[bat_users], rec_item_emb[bat_pos_items], \
                                                       rec_item_emb[bat_neg_items]
                rec_loss = self.bpr_loss(user_emb, pos_item_emb, neg_item_emb)
                cl_loss = self.cl_rate * self.cal_cl_loss([bat_users, bat_pos_items], rec_user_emb, cl_user_emb,
                                                          rec_item_emb, cl_item_emb)
                loss = rec_loss + self.l2_reg_loss(self.reg, user_emb, pos_item_emb) + cl_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))
            # self.logger.info("[iter %d:time:%f]" % (epoch, time.time() - training_start_time))
            # self.logger.info("epoch %d" % (epoch))


    # @timer
    def evaluate_model(self):
        self.XSimGCL.eval()
        return self.evaluator.evaluate(self)

    def predict(self, users):
        users = torch.from_numpy(np.asarray(users)).long().to(self.device)
        return self.XSimGCL.predict(users).cpu().detach().numpy()
