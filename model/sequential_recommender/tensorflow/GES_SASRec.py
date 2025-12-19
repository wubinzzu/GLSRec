"""
Tianyu Zhu, Leilei Sun, and Guoqing Chen.
"Graph-based Embedding Smoothing for Sequential Recommendation."
IEEE Transactions on Knowledge and Data Engineering (2021).
"""
import tensorflow as tf
import numpy as np
from model.base import AbstractRecommender
import scipy.sparse as sp
from util.tensorflow import get_session
from data.dataset import Interaction
import time


class GES(object):
    def __init__(self, adj_matrix, num_user, num_item, args):
        print('model preparing...')
        self.adj_matrix = tf.SparseTensor(adj_matrix[0], adj_matrix[1], adj_matrix[2])
        self.num_user = num_user
        self.num_item = num_item

        self.num_factor = args.num_factor
        self.l2_reg = args.l2_reg
        self.lr = args.lr
        self.max_len = args.max_len
        self.num_block = args.num_block
        self.num_head = args.num_head
        self.num_layer = args.num_layer

        self.emb_dropout_rate = tf.placeholder(tf.float32)
        self.node_dropout_rate = tf.placeholder(tf.float32)
        self.input_seq = tf.placeholder(tf.int32, [None, self.max_len], name='input_seq')
        self.pos_seq = tf.placeholder(tf.int32, [None, self.max_len], name='pos_seq')
        self.neg_seq = tf.placeholder(tf.int32, [None, self.max_len], name='neg_seq')

        mask = tf.expand_dims(tf.to_float(tf.not_equal(self.input_seq, self.num_item)), -1)

        with tf.name_scope('item_embedding'):
            item_embedding = tf.Variable(tf.random_normal([self.num_item, self.num_factor], stddev=0.01, name='item_embedding'))

        with tf.name_scope('graph_convolution'):
            adj_matrix_dropout = self.node_dropout(self.adj_matrix, len(adj_matrix[0]), 1-self.node_dropout_rate)
            item_embedding_final = [item_embedding]
            layer = item_embedding
            if args.gnn == 'gcn':
                W = list()
                for k in range(self.num_layer):
                    W.append(tf.Variable(tf.random_normal([self.num_factor, self.num_factor], stddev=0.01), name='W'+str(k)))
                    # layer = tf.nn.tanh(tf.matmul(tf.sparse_tensor_dense_matmul(adj_matrix_dropout, layer), W[k]))
                    adj_matrix_dropout_dense = tf.sparse_tensor_to_dense(adj_matrix_dropout)
                    layer = tf.nn.tanh(tf.matmul(tf.matmul(adj_matrix_dropout_dense, layer), W[k]))
                    item_embedding_final += [layer]
            elif args.gnn == 'sgc':
                for k in range(self.num_layer):
                    #layer = tf.sparse_tensor_dense_matmul(adj_matrix_dropout, layer)
                    adj_matrix_dropout_dense = tf.sparse_tensor_to_dense(adj_matrix_dropout)
                    layer = tf.matmul(adj_matrix_dropout_dense, layer)
                    item_embedding_final += [layer]
            else:
                pass

        with tf.name_scope('layer_aggregation'):
            if args.layer_agg == 'sum':
                item_embedding_final = tf.reduce_sum(tf.stack(item_embedding_final, 1), 1)
            elif args.layer_agg == 'avg':
                item_embedding_final = tf.reduce_mean(tf.stack(item_embedding_final, 1), 1)
            elif args.layer_agg == 'concat':
                item_embedding_final = tf.concat(item_embedding_final, 1)
                self.num_factor *= (self.num_layer + 1)
            else:
                item_embedding_final = item_embedding_final[-1]
            item_embedding_final_ = tf.concat([item_embedding_final, tf.zeros([1, self.num_factor])], 0)

        with tf.name_scope('positional_embedding'):
            position = tf.tile(tf.expand_dims(tf.range(self.max_len), 0), [tf.shape(self.input_seq)[0], 1])
            position_embedding = tf.Variable(tf.random_normal([self.max_len, self.num_factor], stddev=0.01), name='position_embedding')
            p_emb = tf.nn.embedding_lookup(position_embedding, position)

        with tf.name_scope('dropout'):
            self.seq = tf.nn.embedding_lookup(item_embedding_final_, self.input_seq) * (self.num_factor ** 0.5) + p_emb
            self.seq = tf.nn.dropout(self.seq, keep_prob=1-self.emb_dropout_rate) * mask

        with tf.name_scope('block'):
            for _ in range(self.num_block):
                # Self-attention
                # Linear projections
                seq = self.seq
                seq_norm = self.layer_normalize(seq)
                Q = tf.layers.dense(seq_norm, self.num_factor, activation=None)
                K = tf.layers.dense(seq, self.num_factor, activation=None)
                V = tf.layers.dense(seq, self.num_factor, activation=None)

                # Split and concat
                Q_ = tf.concat(tf.split(Q, self.num_head, axis=2), axis=0)
                K_ = tf.concat(tf.split(K, self.num_head, axis=2), axis=0)
                V_ = tf.concat(tf.split(V, self.num_head, axis=2), axis=0)

                # Multiplication
                outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))

                # Scale
                outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

                # Key Masking
                key_masks = tf.sign(tf.reduce_sum(tf.abs(seq), axis=-1))
                key_masks = tf.tile(key_masks, [self.num_head, 1])
                key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(seq_norm)[1], 1])

                paddings = tf.ones_like(outputs)*(-2**32+1)
                outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

                # Causality (Future blinding)
                diag_vals = tf.ones_like(outputs[0, :, :])
                tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
                # tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense() # for earlier tf versions
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])

                paddings = tf.ones_like(masks)*(-2**32+1)
                outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

                # Activation
                outputs = tf.nn.softmax(outputs)

                # Query Masking
                query_masks = tf.sign(tf.reduce_sum(tf.abs(seq_norm), axis=-1))
                query_masks = tf.tile(query_masks, [self.num_head, 1])
                query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(seq)[1]])
                outputs *= query_masks

                # Dropouts
                outputs = tf.nn.dropout(outputs, keep_prob=1-self.emb_dropout_rate)

                # Weighted sum
                outputs = tf.matmul(outputs, V_)

                # Restore shape
                outputs = tf.concat(tf.split(outputs, self.num_head, axis=0), axis=2)

                # Residual connection
                outputs += seq_norm

                # Layer normalization
                outputs = self.layer_normalize(outputs)

                # Feed forward
                # Layer 1
                outputs_ = tf.layers.dense(outputs, self.num_factor, activation=tf.nn.relu, use_bias=True)
                outputs_ = tf.nn.dropout(outputs_, keep_prob=1-self.emb_dropout_rate)

                # Layer 2
                outputs_ = tf.layers.dense(outputs_, self.num_factor, activation=None, use_bias=True)
                outputs_ = tf.nn.dropout(outputs_, keep_prob=1-self.emb_dropout_rate)

                # Residual connection
                outputs_ += outputs

                self.seq = outputs_ * mask

            self.seq = self.layer_normalize(self.seq)

        with tf.name_scope('train'):
            input_seq_emb = tf.reshape(self.seq, [tf.shape(self.input_seq)[0] * self.max_len, self.num_factor])
            pos_seq_emb = tf.nn.embedding_lookup(item_embedding_final_, tf.reshape(self.pos_seq, [tf.shape(self.input_seq)[0] * self.max_len]))
            neg_seq_emb = tf.nn.embedding_lookup(item_embedding_final_, tf.reshape(self.neg_seq, [tf.shape(self.input_seq)[0] * self.max_len]))
            pos_logits = tf.reduce_sum(pos_seq_emb * input_seq_emb, -1)
            neg_logits = tf.reduce_sum(neg_seq_emb * input_seq_emb, -1)
            target = tf.reshape(tf.to_float(tf.not_equal(self.pos_seq, self.num_item)), [tf.shape(self.input_seq)[0] * self.max_len])
            loss = -tf.reduce_sum(tf.log(tf.sigmoid(pos_logits) + 1e-24) * target + tf.log(1 - tf.sigmoid(neg_logits) + 1e-24) * target) / tf.reduce_sum(target)
            self.loss = loss + self.l2_reg * tf.reduce_sum([tf.nn.l2_loss(va) for va in tf.trainable_variables()])
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.name_scope('test'):
            # self.test_item = tf.placeholder(tf.int32, [None, 1000])
            test_item_emb = item_embedding_final # tf.nn.embedding_lookup(item_embedding_final, self.test_item) # [batch_size, 1000, self.num_factor]
            input_seq_emb_reshape = tf.reshape(input_seq_emb, [tf.shape(self.input_seq)[0], self.max_len, self.num_factor])
            input_seq_emb_last = tf.expand_dims(input_seq_emb_reshape[:, -1, :], 1) # [batch_size, 1, self.num_factor]
            test_logits = tf.matmul(tf.squeeze(input_seq_emb_last, 1), tf.transpose(test_item_emb))
            self.test_logits = test_logits #tf.squeeze(test_logits, 1)

    def layer_normalize(self, inputs):
        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(self.num_factor))
        gamma = tf.Variable(tf.ones(self.num_factor))
        normalized = (inputs - mean) / ((variance + 1e-24) ** 0.5)
        outputs = gamma * normalized + beta
        return outputs

    def node_dropout(self, adj_matrix, num_value, keep_prob):
        noise_shape = [num_value]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(adj_matrix, dropout_mask) * tf.div(1.0, keep_prob)
        return pre_out


def get_adj_matrix(train_dict, rel_dict, num_item, alpha, beta, max_len):
    row_seq = [train_dict[u][-max_len:][n] for u in train_dict for n in range(len(train_dict[u][-max_len:])-1)] + \
              [train_dict[u][-max_len:][n+1] for u in train_dict for n in range(len(train_dict[u][-max_len:])-1)]
    col_seq = [train_dict[u][-max_len:][n+1] for u in train_dict for n in range(len(train_dict[u][-max_len:])-1)] + \
              [train_dict[u][-max_len:][n] for u in train_dict for n in range(len(train_dict[u][-max_len:])-1)]

    row_sem = [i for i in rel_dict for j in rel_dict[i]] + [j for i in rel_dict for j in rel_dict[i]]
    col_sem = [j for i in rel_dict for j in rel_dict[i]] + [i for i in rel_dict for j in rel_dict[i]]

    rel_matrix = sp.coo_matrix(([alpha]*len(row_seq)+[beta]*len(row_sem), (row_seq+row_sem, col_seq+col_sem)), (num_item, num_item)).astype(np.float32) + sp.eye(num_item)
    row_sum = np.array(rel_matrix.sum(1)) + 1e-24
    degree_mat_inv_sqrt = sp.diags(np.power(row_sum, -0.5).flatten())
    rel_matrix_normalized = degree_mat_inv_sqrt.dot(rel_matrix.dot(degree_mat_inv_sqrt)).tocoo()
    indices = np.vstack((rel_matrix_normalized.row, rel_matrix_normalized.col)).transpose()
    values = rel_matrix_normalized.data.astype(np.float32)
    shape = rel_matrix_normalized.shape
    return indices, values, shape


def get_train_data(train_dict, num_item, max_len):
    train_data = list()
    for u in train_dict:
        input_seq = np.ones([max_len], dtype=np.int32) * num_item
        pos_seq = np.ones([max_len], dtype=np.int32) * num_item
        neg_seq = np.ones([max_len], dtype=np.int32) * num_item
        nxt = train_dict[u][-1]
        idx = max_len - 1
        for i in reversed(train_dict[u][:-1]):
            input_seq[idx] = i
            pos_seq[idx] = nxt
            if nxt != num_item:
                neg_seq[idx] = np.random.randint(num_item)
                while neg_seq[idx] == nxt:
                #while neg_seq[idx] in train_dict[u]:
                    neg_seq[idx] = np.random.randint(num_item)
            nxt = i
            idx -= 1
            if idx == -1:
                break
        train_data.append([input_seq, pos_seq, neg_seq])
    return train_data


def get_train_batch(train_data, batch_size):
    train_batch = list()
    np.random.shuffle(train_data)
    i = 0
    while i < len(train_data):
        train_batch.append(np.asarray(train_data[i:i+batch_size]))
        i += batch_size
    return train_batch


def get_feed_dict(model, batch_data, emb_dropout_rate, node_dropout_rate):
    feed_dict = dict()
    feed_dict[model.emb_dropout_rate] = emb_dropout_rate
    feed_dict[model.node_dropout_rate] = node_dropout_rate
    feed_dict[model.input_seq] = batch_data[:, 0]
    feed_dict[model.pos_seq] = batch_data[:, 1]
    feed_dict[model.neg_seq] = batch_data[:, 2]
    return feed_dict


def get_test_data(train_dict, num_user, num_item, max_len):
    test_data = np.full([num_user, max_len], num_item, dtype=np.int32)
    for u, seq_items in train_dict.items():
        seq_items = seq_items[-max_len:]
        test_data[u][-len(seq_items):] = seq_items
    return test_data

    # test_data = list()
    # for u in train_dict:
    #     input_seq = np.ones([max_len], dtype=np.int32) * num_item
    #     idx = max_len - 1
    #     for i in reversed(train_dict[u]):
    #         input_seq[idx] = i
    #         idx -= 1
    #         if idx == -1:
    #             break
    #     item_idx = [test_dict[u]]
    #     for neg in negative_dict[u]:
    #         item_idx.append(neg)
    #     test_data.append(list(input_seq) + item_idx)
    # test_data = np.asarray(test_data)
    # return test_data


def get_feed_dict_test(model, item_seqs):
    feed_dict = dict()
    feed_dict[model.emb_dropout_rate] = 0.0
    feed_dict[model.node_dropout_rate] = 0.0
    feed_dict[model.input_seq] = item_seqs
    return feed_dict


class Conf(object):
    def __init__(self, conf_dict: dict):
        self._conf = conf_dict

    def __getattr__(self, item):
        return self._conf[item]


class GES_SASRec(AbstractRecommender):
    def __init__(self, config):
        super(GES_SASRec, self).__init__(config)
        self.conf = Conf(config)

        self.dataset = self.dataset
        self.num_users = self.dataset.num_users
        self.num_items = self.dataset.num_items
        train_csr = self.dataset.train_data.to_csr_matrix()
        rel_csr = train_csr.T.dot(train_csr).tocsr()
        rel_dict = {i: {j: [0] for j in rel_csr[i].indices} for i in range(self.num_items)}
        # self.train_dict = self.dataset.get_user_train_dict(by_time=True)
        self.train_dict = self.dataset.train_data.to_user_dict(by_time=True)
        # self.test_dict = self.dataset.get_user_test_dict()
        self.test_dict = self.dataset.test_data.to_user_dict(by_time=True)
        self.test_data = get_test_data(self.train_dict, self.num_users, self.num_items, self.conf.max_len)
        self.sess = get_session(config["gpu_mem"])
        adj_matrix = get_adj_matrix(self.train_dict, rel_dict, self.num_items,
                                    self.conf.alpha, self.conf.beta, self.conf.max_len)

        self.model = GES(adj_matrix, self.num_users, self.num_items, self.conf)



    def build_graph(self):
        pass

    def train_model(self):
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.logger.info(self.evaluator.metrics_info())
        # training_start_time = time.time();
        for epoch in range(1, self.conf.num_epoch + 1):
            train_data = get_train_data(self.train_dict, self.num_items, self.conf.max_len)
            train_batch = get_train_batch(train_data, self.conf.batch_size)
            for batch in train_batch:
                loss, _ = self.sess.run([self.model.loss, self.model.train_op],
                                        feed_dict=get_feed_dict(self.model, batch, self.conf.emb_dropout_rate,
                                                                self.conf.node_dropout_rate))

            self.logger.info("epoch %d:\t%s" % (epoch, self.evaluate()))
            # self.logger.info("[iter %d:time:%f]" % (epoch, time.time() - training_start_time))

    def evaluate(self):
        return self.evaluator.evaluate(self)
            
    def predict(self, user_ids):
        ratings = []
        # if candidate_items_userids is not None:
        #     print("1233")
        # else:
        ratings = self.sess.run(self.model.test_logits, feed_dict=get_feed_dict_test(self.model, self.test_data[user_ids]))
        return ratings
