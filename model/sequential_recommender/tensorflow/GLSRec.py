import numpy as np
import scipy.sparse as sp
from model.base.abstract_recommender import AbstractRecommender
from data.data_iterator import DataIterator
from reckit import timer
from util.common.tool import csr_to_user_dict_bytime
from collections import defaultdict
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from util.tensorflow import l2_distance
from util.common.tool import batch_randint_choice
from util.tensorflow.func import normalize_adj_matrix
from util.tensorflow import inner_product
from util.tensorflow import get_session
from util.common.tool import csr_to_time_dict
import time
import math


def GELU (x):
    cdf = 0.5*(1.0+tf.tanh((np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x,3))))) 
    return x*cdf

def  LayerNorm(self,inputs, epsilon=1e-8, scope="ln", reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keepdims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta
    return outputs

def mexp(x, tau=1.0):
    # normalize att_logit to avoid negative value
    x_max = tf.reduce_max(x)
    x_min = tf.reduce_min(x)
    norm_x = (x-x_min) / (x_max-x_min)
    exp_x = tf.exp(norm_x/tau)
    return exp_x

class GLSRec(AbstractRecommender):
    def __init__(self, conf):
        super(GLSRec, self).__init__(conf)
        dataset = self.dataset
        self.users_num, self.items_num = dataset.train_matrix.shape
        self.lr = conf["lr"]
        self.l2_reg = conf["l2_reg"]
        self.l2_regW = conf["l2_regW"]
        self.factors_num = conf["factors_num"]
        self.seq_L = conf["seq_L"]
        self.neg_samples = conf["neg_samples"]
        self.batch_size = conf["batch_size"]
        self.epochs = conf["epochs"]
        self.num_blocks = conf["num_blocks"]
        # GCN's hyperparameters
        self.n_layers = conf['n_layers']
        self.n_layers_ii = conf["n_layers_ii"]
        self.norm_adj = self.create_adj_mat()

        self.user_pos_train = self.dataset.train_data.to_user_dict(by_time=True)
        self.user_pos_time = csr_to_time_dict(self.dataset.time_matrix)  # pos time

        self.sess = get_session(conf["gpu_mem"])
        self.build_graph()
        self.sess.run(tf.global_variables_initializer())
    
    @timer
    def create_adj_mat(self):
        user_list, item_list, _ = self.dataset.get_train_interactionssecond()
        user_np = np.array(user_list, dtype=np.int32)
        item_np = np.array(item_list, dtype=np.int32)
        ratings = np.ones_like(user_np, dtype=np.float32)
        n_nodes = self.users_num + self.items_num
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.users_num)), shape=(n_nodes, n_nodes))
        adj_mat = tmp_adj + tmp_adj.T

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix

    def _create_gcn_embed(self):
        adj_mat = self._convert_sp_mat_to_sp_tensor(self.norm_adj)

        ego_embeddings = tf.concat([self.embeddings["user_embeddings"], self.embeddings["item_embeddings"]], axis=0)

        all_embeddings = [ego_embeddings]

        for k in range(0, self.n_layers):
            side_embeddings = tf.sparse_tensor_dense_matmul(adj_mat, ego_embeddings, name="sparse_dense")

            # transformed sum messages of neighbors.
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.users_num, self.items_num], 0)
        return u_g_embeddings, i_g_embeddings

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _create_placeholder(self):
        self.user_ph = tf.placeholder(tf.int32, [None], name="user")
        self.item_seq_ph = tf.placeholder(tf.int32, [None, self.seq_L], name="item_seq")
        self.item_pos_ph = tf.placeholder(tf.int32, [None, ], name="item_pos")
        self.item_neg_ph = tf.placeholder(tf.int32, [None, self.neg_samples], name="item_neg")


    def _construct_graph(self):

        th_rs_dict = defaultdict(list)
        th_rs_dict_time = defaultdict(list)
        for user, pos_items in self.user_pos_train.items():
            seq_times = self.user_pos_time[user]
            seq_times = sorted(seq_times)
            seq_times_set = sorted(set(seq_times))
            seq_times_inter = []
            time_min = 1
            if len(seq_times_set) > 1:
                for i in range(len(seq_times_set) - 1):
                    time_min_tmp = seq_times_set[i + 1] - seq_times_set[i]
                    seq_times_inter.append(time_min_tmp)
                time_min = sorted(seq_times_inter)[0]
            time_i = 0
            for h, t in zip(pos_items[:-1], pos_items[1:]):

                time_ii = math.log((seq_times[time_i + 1] - seq_times[time_i]) / time_min + 1)
                th_rs_dict[(t, h)].append(user)
                th_rs_dict_time[(t, h)].append(time_ii)
                time_i += 1

        th_rs_list = sorted(th_rs_dict.items(), key=lambda x: x[0])
        th_rs_list_time = sorted(th_rs_dict_time.items(), key=lambda x: x[0])

        user_list, head_list, tail_list, time_distance_list = [], [], [], []
        for (t, h), r in th_rs_list:
            user_list.extend(r)
            head_list.extend([h] * len(r))
            tail_list.extend([t] * len(r))
        for (t, h), time in th_rs_list_time:
            time_distance_list.extend(time)


        row_idx, nnz = np.unique(tail_list, return_counts=True)
        count = {r: n for r, n in zip(row_idx, nnz)}
        nnz = [count[i] if i in count else 0 for i in range(self.items_num)]
        nnz = np.concatenate([[0], nnz])
        rows_idx = np.cumsum(nnz)


        edge_num = np.array([len(r) for (t, h), r in th_rs_list], dtype=np.int32)
        edge_num = np.concatenate([[0], edge_num])
        edge_idx = np.cumsum(edge_num)

        sp_idx = [[t, h] for (t, h), r in th_rs_list]
        adj_mean_norm = self._get_mean_norm(edge_num[1:], sp_idx)

        return head_list, tail_list, user_list, rows_idx, edge_idx, sp_idx, adj_mean_norm, time_distance_list

    @timer
    def _init_constant(self):

        head_list, tail_list, user_list, rows_idx, edge_idx, sp_idx, adj_norm, time_distance_list = self._construct_graph()


        self.att_head_idx = tf.constant(head_list, dtype=tf.int32, shape=None, name="att_head_idx")
        self.att_tail_idx = tf.constant(tail_list, dtype=tf.int32, shape=None, name="att_tail_idx")
        self.att_user_idx = tf.constant(user_list, dtype=tf.int32, shape=None, name="att_user_idx")
        self.att_time_idx = tf.constant(time_distance_list, dtype=float)
        self.att_time_idx = tf.expand_dims(self.att_time_idx, 1)

        self.rows_end_idx = tf.constant(rows_idx[1:], dtype=tf.int32, shape=None, name="rows_end_idx")
        self.row_begin_idx = tf.constant(rows_idx[:-1], dtype=tf.int32, shape=None, name="row_begin_idx")


        self.edge_end_idx = tf.constant(edge_idx[1:], dtype=tf.int32, shape=None, name="edge_end_idx")
        self.edge_begin_idx = tf.constant(edge_idx[:-1], dtype=tf.int32, shape=None, name="edge_begin_idx")


        self.sp_tensor_idx = tf.constant(sp_idx, dtype=tf.int64)
        self.adj_norm = None

    def _get_mean_norm(self, edge_num, sp_idx):
        adj_num = np.array(edge_num, dtype=np.float32)
        rows, cols = list(zip(*sp_idx))
        adj_mat = sp.csr_matrix((adj_num, (rows, cols)), shape=(self.items_num, self.items_num))

        return normalize_adj_matrix(adj_mat, "left").astype(np.float32)

    def _item_gcn(self, item_emb_ii, user_emb):
        with tf.name_scope("item_gcn"):
            item_emb = item_emb_ii
            for k in range(self.n_layers_ii):
                att_scores = self._get_attention(item_emb, user_emb)
                neighbor_embeddings = tf.sparse_tensor_dense_matmul(att_scores, item_emb)
                item_emb = item_emb + neighbor_embeddings
            return item_emb

    def _get_attention(self, item_embeddings, user_embeddings):

        h_embed = tf.nn.embedding_lookup(item_embeddings, self.att_head_idx)
        r_embed = tf.nn.embedding_lookup(user_embeddings, self.att_user_idx)
        t_embed = tf.nn.embedding_lookup(item_embeddings, self.att_tail_idx)
        att_logit = l2_distance(h_embed+r_embed, t_embed)
        exp_logit = mexp(-att_logit, 1.0)
        exp_logit = tf.concat([[0], exp_logit], axis=0)
        sum_exp_logit = tf.cumsum(exp_logit)
        pre_sum = tf.gather(sum_exp_logit, self.edge_begin_idx)
        next_sum = tf.gather(sum_exp_logit, self.edge_end_idx)
        sum_exp_logit_per_edge = next_sum - pre_sum

        exp_logit = tf.SparseTensor(indices=self.sp_tensor_idx, values=sum_exp_logit_per_edge,
                                    dense_shape=[self.items_num, self.items_num])

        next_sum = tf.gather(sum_exp_logit, self.rows_end_idx)
        pre_sum = tf.gather(sum_exp_logit, self.row_begin_idx)
        sum_exp_logit_per_row = next_sum - pre_sum + 1e-6
        sum_exp_logit_per_row = tf.reshape(sum_exp_logit_per_row, shape=[self.items_num, 1])
        att_score = exp_logit / sum_exp_logit_per_row
        return att_score

    def _create_variable(self):
        self.weights = dict()
        self.embeddings = dict()

        embeding_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        user_embeddings = tf.Variable(embeding_initializer([self.users_num, self.factors_num]), dtype=tf.float32)
        self.embeddings.setdefault("user_embeddings", user_embeddings)

        q_item_embeddings = tf.Variable(embeding_initializer([self.items_num, self.factors_num]), dtype=tf.float32)
        self.embeddings.setdefault("q_item_embeddings", q_item_embeddings)

        item_embeddings = tf.Variable(embeding_initializer([self.items_num, self.factors_num]), dtype=tf.float32)
        self.embeddings.setdefault("item_embeddings", item_embeddings)

        position_embeddings = tf.Variable(embeding_initializer([self.seq_L, self.factors_num]), dtype=tf.float32)
        self.embeddings.setdefault("position_embeddings", position_embeddings)

        target_item_embeddings = tf.Variable(embeding_initializer([self.items_num, self.factors_num]), dtype=tf.float32)
        self.embeddings.setdefault("target_item_embeddings", target_item_embeddings)

        item_biases = tf.Variable(embeding_initializer([self.items_num]), dtype=tf.float32)
        self.embeddings.setdefault("item_biases", item_biases)

        self.user_embeddings, item_embs = self._create_gcn_embed()
        self.q_item_embs = self._item_gcn(item_embeddings, self.user_embeddings)
        zero_pad = tf.zeros([1, self.factors_num], name="padding2")
        self.item_embeddings = tf.concat([item_embs, zero_pad], axis=0)
        self.q_item_embs = tf.concat([q_item_embeddings, zero_pad], axis=0)

        Gate_initializer = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
        self.weights.setdefault("mlp1_w1",
                                tf.Variable(Gate_initializer([self.seq_L, 4*self.seq_L]), dtype=tf.float32))

        self.weights.setdefault("mlp1_w2",
                                tf.Variable(Gate_initializer([self.seq_L, 4*self.seq_L]), dtype=tf.float32))
        
        self.weights.setdefault("mlp1_w3",
                                tf.Variable(Gate_initializer([4*self.seq_L, self.seq_L]), dtype=tf.float32))
        
        
        self.weights.setdefault("mlp2_w1",
                                tf.Variable(Gate_initializer([self.factors_num, 4*self.factors_num]), dtype=tf.float32))
        
        self.weights.setdefault("mlp2_w2",
                                tf.Variable(Gate_initializer([self.factors_num, 4*self.factors_num]), dtype=tf.float32))
        
        self.weights.setdefault("mlp2_w3",
                                tf.Variable(Gate_initializer([4*self.factors_num, self.factors_num]), dtype=tf.float32))



    def _create_inference(self):

        self.batch_size_b = tf.shape(self.item_seq_ph)[0]

        self.user_embs = tf.nn.embedding_lookup(self.user_embeddings, self.user_ph)
        
        user_embs = tf.expand_dims(self.user_embs, axis=1)

        self.position_embedings = tf.tile(tf.expand_dims(self.embeddings["position_embeddings"], 0), tf.stack([self.batch_size_b,1, 1]))

        self.item_embs = tf.nn.embedding_lookup(self.q_item_embs, self.item_seq_ph)  # b,L,D

        position = tf.tile(tf.expand_dims(tf.range(tf.shape(self.item_seq_ph)[1]), 0),
                           [tf.shape(self.item_seq_ph)[0], 1])
        t = tf.nn.embedding_lookup(self.embeddings["position_embeddings"], position)
        self.item_embs +=t
        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.item_seq_ph, self.items_num)), -1)
        self.item_embs *= mask

        embedding_Pu = tf.tile(user_embs, tf.stack([1, self.seq_L, 1]))
        embedding_Pu_T = tf.transpose(embedding_Pu, [0, 2, 1])
        for _ in range(self.num_blocks):
            factor_input = tf.transpose(self.item_embs, [0, 2, 1])

            factor_mixer = tf.matmul(tf.reshape(embedding_Pu_T, [-1, self.seq_L]), self.weights["mlp1_w1"]) + \
                          tf.matmul(tf.reshape(factor_input, [-1, self.seq_L]), self.weights["mlp1_w2"])
            factor_mixer = GELU(factor_mixer)
            factor_mixer = tf.matmul(factor_mixer, self.weights["mlp1_w3"]) 
            factor_mixer = tf.reshape(factor_mixer, [-1, self.factors_num, self.seq_L])

            factor_mixer = factor_mixer + factor_input

            item_input = tf.transpose(factor_mixer, [0, 2, 1])

            item_mixer = tf.matmul(tf.reshape(embedding_Pu, [-1, self.factors_num]), self.weights["mlp2_w1"]) + \
                        tf.matmul(tf.reshape(item_input, [-1, self.factors_num]), self.weights["mlp2_w2"])
            
            item_mixer = GELU(item_mixer)
            item_mixer = tf.matmul(item_mixer, self.weights["mlp2_w3"])
            
            item_mixer = tf.reshape(item_mixer, [-1, self.seq_L, self.factors_num])

            item_mixer = item_mixer + item_input

            self.item_embs = item_mixer * mask

        self.short_embeddings = tf.reduce_sum(self.item_embs, axis=1)
        

        self.output = self.short_embeddings

        return self.output

    def _create_loss(self):
        self.user_embs = tf.nn.embedding_lookup(self.user_embeddings, self.user_ph)

        user_embs = tf.expand_dims(self.user_embs, axis=1)
        self.p = self.user_embs + self._create_inference()
        
        self.tar_item_emb_pos = tf.nn.embedding_lookup(self.q_item_embs, self.item_pos_ph)

        self.tar_item_emb_neg = tf.nn.embedding_lookup(self.q_item_embs, self.item_neg_ph)
        
        alpha = tf.random.uniform(tf.shape(self.tar_item_emb_neg), minval=0, maxval=1)
        
        neg_emb = tf.multiply(alpha, tf.expand_dims(self.p, axis=1)) +tf.multiply((tf.ones_like(alpha)-alpha),self.tar_item_emb_neg)
        
        scores = tf.reduce_sum(tf.multiply(tf.expand_dims(self.p, axis=1), neg_emb), axis=-1)
        
        
        indices = tf.argmax(scores,axis=1)
        self.chosen_neg_emb = tf.gather(neg_emb, indices,axis=1)

        pos_scores = inner_product(self.p,self.tar_item_emb_pos)
        neg_scores = inner_product(self.p, self.chosen_neg_emb)
        loss = tf.reduce_mean(tf.nn.softplus(-(pos_scores - neg_scores)))
        
        self.L2_emb = tf.reduce_sum(tf.square(self.user_embs)) + tf.reduce_sum(tf.square(self.tar_item_emb_pos)) + \
                      tf.reduce_sum(tf.square(self.tar_item_emb_neg)) + tf.reduce_sum(tf.square(self.item_embs)) + \
                      tf.reduce_sum(tf.square(self.position_embedings))

        self.L2_weight = tf.reduce_sum(tf.square(self.weights["mlp1_w1"])) + \
                         tf.reduce_sum(tf.square(self.weights["mlp1_w2"])) + \
                         tf.reduce_sum(tf.square(self.weights["mlp1_w3"])) + \
                         tf.reduce_sum(tf.square(self.weights["mlp2_w1"])) + \
                         tf.reduce_sum(tf.square(self.weights["mlp2_w2"]))+\
                         tf.reduce_sum(tf.square(self.weights["mlp2_w3"]))

        self.Loss_0 = loss + self.l2_reg * self.L2_emb + self.l2_regW * self.L2_weight

        self.all_logits = tf.matmul(self.user_embs, self.item_embeddings,
                                    transpose_b=True) + tf.matmul(self.p,self.q_item_embs,transpose_b=True)

    def _create_optimizer(self):
        self.train_opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.Loss_0)

    def build_graph(self):
        self._create_placeholder()
        self._init_constant()
        self._create_variable()
        self._create_loss()
        self._create_optimizer()

    def train_model(self):

        users_list, item_seq_list, item_pos_list = self._generate_sequences()
        training_start_time = time.time()
        self.logger.info(self.evaluator.metrics_info())
        for epoch in range(self.epochs):
            item_neg_list = self._sample_negative(users_list)
            data = DataIterator(users_list, item_seq_list, item_pos_list, item_neg_list,
                                batch_size=self.batch_size, shuffle=True)
            for bat_user, bat_item_seq, bat_item_pos, bat_item_neg in data:
                feed = {self.user_ph: bat_user,
                        self.item_seq_ph: bat_item_seq,
                        self.item_pos_ph: bat_item_pos,
                        self.item_neg_ph: bat_item_neg}

                self.sess.run(self.train_opt, feed_dict=feed)

            result = self.evaluate_model()
            self.logger.info("epoch %d:\t%s" % (epoch, result))
            # self.logger.info("[iter %d:time:%f]" % (epoch, time.time() - training_start_time))

    def _generate_sequences(self):
        self.user_test_seq = {}
        user_list, item_seq_list, item_pos_list = [], [], []
        userid_set = np.unique(list(self.user_pos_train.keys()))

        for user_id in userid_set:
            seq_items = self.user_pos_train[user_id]

            for index_id in range(len(seq_items)):
                if index_id < 1:
                    continue

                content_data = list()

                for cindex in range(max([0, index_id - self.seq_L]), index_id):
                    content_data.append(seq_items[cindex])

                if len(content_data) < self.seq_L:
                    content_data = content_data + [self.items_num for _ in range(self.seq_L - len(content_data))]

                user_list.append(user_id)
                item_seq_list.append(content_data)
                item_pos_list.append(seq_items[index_id])

            user_id_seq = seq_items[-min([len(seq_items), self.seq_L]):]

            user_id_seq = list(user_id_seq)

            if len(user_id_seq) < self.seq_L:
                padding_length = self.seq_L - len(user_id_seq)
                user_id_seq = user_id_seq + [self.items_num] * padding_length

            self.user_test_seq[user_id] = user_id_seq

        return user_list, item_seq_list, item_pos_list

    def _sample_negative(self, users_list):
        neg_items_list = []
        user_neg_items_dict = {}
        all_userids, all_counts = np.unique(users_list, return_counts=True)
        user_count = DataIterator(all_userids, all_counts, batch_size=1024, shuffle=False)
        for bat_users, bat_counts in user_count:
            n_neg_items = [c * self.neg_samples for c in bat_counts]
            exclusion = [self.user_pos_train[u] for u in bat_users]
            bat_neg = batch_randint_choice(self.items_num, n_neg_items, replace=True, exclusion=exclusion)
            for u, neg in zip(bat_users, bat_neg):
                user_neg_items_dict[u] = neg

        for u, c in zip(all_userids, all_counts):
            neg_items = np.reshape(user_neg_items_dict[u], newshape=[c, self.neg_samples])
            neg_items_list.extend(neg_items)
        return neg_items_list

    def evaluate_model(self):
        return self.evaluator.evaluate(self)

    def predict(self, users, items=None):
        users = DataIterator(users, batch_size=512, shuffle=False, drop_last=False)
        all_ratings = []
        for bat_user in users:
            bat_seq = [self.user_test_seq[u] for u in bat_user]
            feed = {self.user_ph: bat_user,
                    self.item_seq_ph: bat_seq
                    }
            bat_ratings = self.sess.run(self.all_logits, feed_dict=feed)
            all_ratings.extend(bat_ratings)
        all_ratings = np.array(all_ratings, dtype=np.float32)

        if items is not None:
            all_ratings = [all_ratings[idx][item] for idx, item in enumerate(items)]

        return all_ratings