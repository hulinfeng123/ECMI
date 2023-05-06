import datetime
from base.graphRecommender import GraphRecommender
from base.socialRecommender import SocialRecommender
import tensorflow as tf
from scipy.sparse import coo_matrix, eye
import scipy.sparse as sp
import numpy as np
import os
from util import config
from util.loss import bpr_loss
import random

config_gpu = tf.compat.v1.ConfigProto()
config_gpu.gpu_options.allow_growth = True
config_gpu.allow_soft_placement = True
config_gpu.log_device_placement = True
config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.9
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class ECMI(SocialRecommender, GraphRecommender):
    def __init__(self, conf, trainingSet=None, testSet=None, relation=None, itemRelation=None, fold='[1]'):
        GraphRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, fold=fold)
        SocialRecommender.__init__(self, conf=conf, trainingSet=trainingSet, testSet=testSet, relation=relation,
                                   itemRelation=itemRelation, fold=fold)
        # self.itemRelation = itemRelation

    def readConfiguration(self):
        super(ECMI, self).readConfiguration()
        args = config.OptionConf(self.config['ECMI'])
        self.n_layers = int(args['-n_layer'])
        self.ssl_temp = float(args['-temperature'])

        self.ss_rate = float(args['-ss_rate'])
        self.ss_item_rate = float(args['-ss_item_rate'])
        self.struct_rate = float(args['-struct_rate'])
        self.alpha = float(args['-alpha'])

        self.drop_rate = float(args['-drop_rate'])
        self.instance_cnt = int(args['-ins_cnt'])
        self.hyper_layers = int(self.config['hyper_layers'])

    def buildSparseRatingMatrix(self):
        row, col, entries = [], [], []
        count = 0
        for pair in self.data.trainingData:
            # symmetric matrix
            row += [self.data.user[pair[0]]]
            col += [self.data.item[pair[1]]]
            entries += [1.0]
        ratingMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users, self.num_items), dtype=np.float32)
        return ratingMatrix

    def buildSparseRelationMatrix(self):
        row, col, entries = [], [], []
        for pair in self.social.relation:
            # symmetric matrix
            row += [self.data.user[pair[0]]]
            col += [self.data.user[pair[1]]]
            entries += [1.0]
        AdjacencyMatrix = coo_matrix((entries, (row, col)), shape=(self.num_users,self.num_users),dtype=np.float32)
        return AdjacencyMatrix

    def get_complex_social_matrix(self):
        S = self.buildSparseRelationMatrix()
        B = S.multiply(S.T)
        U = S - B
        C1 = (U.dot(U)).multiply(U.T)
        A1 = C1 + C1.T
        C2 = (B.dot(U)).multiply(U.T) + (U.dot(B)).multiply(U.T) + (U.dot(U)).multiply(B)
        A2 = C2 + C2.T
        C3 = (B.dot(B)).multiply(U) + (B.dot(U)).multiply(B) + (U.dot(B)).multiply(B)
        A3 = C3 + C3.T
        A4 = (B.dot(B)).multiply(B)
        C5 = (U.dot(U)).multiply(U) + (U.dot(U.T)).multiply(U) + (U.T.dot(U)).multiply(U)
        A5 = C5 + C5.T
        A6 = (U.dot(B)).multiply(U) + (B.dot(U.T)).multiply(U.T) + (U.T.dot(U)).multiply(B)
        A7 = (U.T.dot(B)).multiply(U.T) + (B.dot(U)).multiply(U) + (U.dot(U.T)).multiply(B)
        H_s = sum([A1, A2, A3, A4, A5, A6, A7])
        H_s = H_s.multiply(1.0 / H_s.sum(axis=1).reshape(-1, 1))
        return H_s


    def get_birectional_social_matrix(self):
        row_idx = [self.data.user[pair[0]] for pair in
                   self.social.relation]
        col_idx = [self.data.user[pair[1]] for pair in self.social.relation]
        follower_np = np.array(row_idx)  # 25312
        followee_np = np.array(col_idx)
        relations = np.ones_like(follower_np, dtype=np.float32)  # 25312
        tmp_adj = sp.csr_matrix((relations, (follower_np, followee_np)),
                                shape=(self.num_users, self.num_users))  # 1889 * 1889
        adj_mat = tmp_adj.multiply(tmp_adj)  # 1889 * 1889
        return adj_mat

    def get_birectional_item_matrix(self):
        row_idx = [self.data.item[pair[0]] for pair in self.item.relation]
        col_idx = [self.data.item[pair[1]] for pair in self.item.relation]
        follower_np = np.array(row_idx)
        followee_np = np.array(col_idx)
        relations = np.ones_like(follower_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((relations, (follower_np, followee_np)), shape=(self.num_items, self.num_items))
        adj_mat = tmp_adj.multiply(tmp_adj)  # 1889 * 1889
        return adj_mat

    def get_social_related_views(self, social_mat, rating_mat):
        def normalization(M):
            rowsum = np.array(M.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(M)
            return norm_adj_tmp.dot(d_mat_inv)

        social_matrix = social_mat.dot(social_mat)
        social_matrix = social_matrix.multiply(social_mat) + eye(self.num_users)
        sharing_matrix = rating_mat.dot(rating_mat.T)
        # print("=======user======")
        # print(sharing_matrix.shape)
        # print(social_mat.shape)
        # print(eye(self.num_users).shape)
        # print("=======user======")
        sharing_matrix = sharing_matrix.multiply(social_mat) + eye(self.num_users)
        social_matrix = normalization(social_matrix)
        sharing_matrix = normalization(sharing_matrix)
        return [social_matrix, sharing_matrix]

    def get_item_related_views(self, bs_item_matrix, rating_mat):
        def normalization(M):
            rowsum = np.array(M.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj_tmp = d_mat_inv.dot(M)
            return norm_adj_tmp.dot(d_mat_inv)

        attribute_mat = bs_item_matrix.dot(bs_item_matrix)
        attribute_mat = attribute_mat.multiply(bs_item_matrix) + eye(self.num_items)
        # latent_matrix = rating_mat.dot(rating_mat.T)
        latent_matrix = rating_mat.T.dot(rating_mat)
        # print("=======item======")
        # print(latent_matrix.shape)
        # print(bs_item_matrix.shape)
        # print(eye(self.num_items).shape)
        # print("=======item======")
        latent_matrix = latent_matrix.multiply(bs_item_matrix) + eye(self.num_items)
        attribute_mat = normalization(attribute_mat)
        latent_matrix = normalization(latent_matrix)
        return [attribute_mat, latent_matrix]

    def _create_variable(self):
        self.sub_social_mat = {}
        self.sub_item_mat = {}

        self.sub_social_mat['adj_values_sub'] = tf.compat.v1.placeholder(tf.float32)
        self.sub_social_mat['adj_indices_sub'] = tf.compat.v1.placeholder(tf.int64)
        self.sub_social_mat['adj_shape_sub'] = tf.compat.v1.placeholder(tf.int64)
        self.sub_social_mat['sub_social_mat'] = tf.SparseTensor(
            self.sub_social_mat['adj_indices_sub'],
            self.sub_social_mat['adj_values_sub'],
            self.sub_social_mat['adj_shape_sub'])

        self.sub_item_mat['adj_values_sub'] = tf.compat.v1.placeholder(tf.float32)
        self.sub_item_mat['adj_indices_sub'] = tf.compat.v1.placeholder(tf.int64)
        self.sub_item_mat['adj_shape_sub'] = tf.compat.v1.placeholder(tf.int64)
        self.sub_item_mat['sub_item_mat'] = tf.SparseTensor(
            self.sub_item_mat['adj_indices_sub'],
            self.sub_item_mat['adj_values_sub'],
            self.sub_item_mat['adj_shape_sub'])



    def get_social_adj_mat(self, is_subgraph=False):
        n_nodes = self.num_users + self.num_items
        row_idx = [self.data.user[pair[0]] for pair in self.data.trainingData]
        col_idx = [self.data.item[pair[1]] for pair in self.data.trainingData]
        s_row_idx = [self.data.user[pair[0]] for pair in self.social.relation]
        s_col_idx = [self.data.user[pair[1]] for pair in self.social.relation]

        if is_subgraph and self.drop_rate > 0:
            keep_idx = random.sample(list(range(self.data.elemCount())),
                                     int(self.data.elemCount() * (1 - self.drop_rate)))
            user_np = np.array(row_idx)[keep_idx]
            item_np = np.array(col_idx)[keep_idx]
            ratings = np.ones_like(user_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (user_np, self.num_users + item_np)), shape=(n_nodes, n_nodes))
            adj_mat = tmp_adj + tmp_adj.T
            skeep_idx = random.sample(list(range(len(s_row_idx))), int(len(s_row_idx) * (1 - self.drop_rate)))

            follower_np = np.array(s_row_idx)[skeep_idx]
            followee_np = np.array(s_col_idx)[skeep_idx]

            relations = np.ones_like(follower_np, dtype=np.float32)

            social_mat = sp.csr_matrix((relations, (follower_np, followee_np)), shape=(n_nodes, n_nodes))
            social_mat = social_mat.multiply(social_mat)

            adj_social_mat = adj_mat + social_mat
        # else:
        #     user_np = np.array(row_idx)
        #     item_np = np.array(col_idx)
        #     ratings = np.ones_like(user_np, dtype=np.float32)
        #     tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_users)), shape=(n_nodes, n_nodes))
        #     adj_mat = tmp_adj + tmp_adj.T
        # pre adjcency matrix
        # adj_matrix = D^(-1/2)·A·D^(-1/2)
        adj_social_mat = self.get_pre_adj_matrix(adj_social_mat)
        return adj_social_mat

    def get_item_adj_mat(self, is_subgraph=False):
        n_nodes = self.num_users + self.num_items
        row_idx = [self.data.user[pair[0]] for pair in self.data.trainingData]
        col_idx = [self.data.item[pair[1]] for pair in self.data.trainingData]

        s_row_idx_item = [self.data.item[pair[0]] for pair in self.item.relation]
        s_col_idx_item = [self.data.item[pair[1]] for pair in self.item.relation]

        if is_subgraph and self.drop_rate > 0:
            keep_idx = random.sample(list(range(self.data.elemCount())),
                                     int(self.data.elemCount() * (1 - self.drop_rate)))
            user_np = np.array(row_idx)[keep_idx]
            item_np = np.array(col_idx)[keep_idx]
            ratings = np.ones_like(user_np, dtype=np.float32)
            tmp_adj = sp.csr_matrix((ratings, (user_np, self.num_users + item_np)), shape=(n_nodes, n_nodes))
            adj_mat = tmp_adj + tmp_adj.T

            skeep_idx_item = random.sample(list(range(len(s_row_idx_item))), int(len(s_row_idx_item) * (1 - self.drop_rate)))

            follower_np_item = np.array(s_row_idx_item)[skeep_idx_item]
            followee_np_item = np.array(s_col_idx_item)[skeep_idx_item]

            relations_item = np.ones_like(follower_np_item, dtype=np.float32)

            item_mat = sp.csr_matrix((relations_item, (follower_np_item, followee_np_item)), shape=(n_nodes, n_nodes))
            item_mat = item_mat.multiply(item_mat)

            adj_item_mat = adj_mat + item_mat
        # else:
        #     user_np = np.array(row_idx)
        #     item_np = np.array(col_idx)
        #     ratings = np.ones_like(user_np, dtype=np.float32)
        #     tmp_adj = sp.csr_matrix((ratings, (user_np, item_np + self.num_users)), shape=(n_nodes, n_nodes))
        #     adj_mat = tmp_adj + tmp_adj.T
        # pre adjcency matrix
        # adj_matrix = D^(-1/2)·A·D^(-1/2)
        adj_item_mat = self.get_pre_adj_matrix(adj_item_mat)
        return adj_item_mat

    
    def get_pre_adj_matrix(self, adj_mat):
        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        return adj_matrix


    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _convert_csr_to_sparse_tensor_inputs(self, X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return indices, coo.data, coo.shape

    def initModel(self):
        super(ECMI, self).initModel()
        self.neg_idx = tf.compat.v1.placeholder(tf.int32, name="neg_holder")
        self._create_variable()
        self.bs_matrix = self.get_birectional_social_matrix()  # 1890 * 1890
        self.bs_item_matrix = self.get_birectional_item_matrix()
        self.rating_mat = self.buildSparseRatingMatrix()  # 1890 * 15413

        social_mat, sharing_mat = self.get_social_related_views(self.bs_matrix, self.rating_mat)

        sharing_mat = self._convert_sp_mat_to_sp_tensor(sharing_mat)
        complex_mat = self.get_complex_social_matrix()
        complex_social_mat = self._convert_sp_mat_to_sp_tensor(complex_mat)

        attribute_mat, latent_mat = self.get_item_related_views(self.bs_item_matrix, self.rating_mat)
        attribute_mat = self._convert_sp_mat_to_sp_tensor(attribute_mat)
        latent_mat = self._convert_sp_mat_to_sp_tensor(latent_mat)

        self.user_embeddings = tf.Variable(
            tf.compat.v1.truncated_normal(shape=[self.num_users, self.emb_size], stddev=0.005),
            name='U') / 2
        self.item_embeddings = tf.Variable(
            tf.compat.v1.truncated_normal(shape=[self.num_items, self.emb_size], stddev=0.005),
            name='V') / 2
        # initialize adjacency matrices
        ui_mat = self.create_joint_sparse_adj_tensor()

        # friend_view_embeddings = self.user_embeddings
        sharing_view_embeddings = self.user_embeddings
        # all_social_embeddings = [friend_view_embeddings]
        all_sharing_embeddings = [sharing_view_embeddings]
        complex_view_embeddings = self.user_embeddings
        all_complex_embeddings = [complex_view_embeddings]

        attribute_view_embeddings = self.item_embeddings
        latent_view_embeddings = self.item_embeddings
        all_attribute_embeddings = [attribute_view_embeddings]
        all_latent_embeddings = [latent_view_embeddings]

        ego_embeddings = tf.concat([self.user_embeddings, self.item_embeddings], axis=0)
        all_embeddings = [ego_embeddings]
        origin_embeddings_list = [ego_embeddings]
        aug_social_embeddings = ego_embeddings
        all_aug_social_embeddings = [ego_embeddings]
        aug_item_embeddings = ego_embeddings
        all_aug_item_embeddings = [ego_embeddings]

        # multi-view convolution: LightGCN structure
        for k in range(self.n_layers):

            sharing_view_embeddings = tf.compat.v1.sparse_tensor_dense_matmul(sharing_mat, sharing_view_embeddings)
            norm_embeddings = tf.math.l2_normalize(sharing_view_embeddings, axis=1)
            all_sharing_embeddings += [norm_embeddings]
            complex_view_embeddings = tf.compat.v1.sparse_tensor_dense_matmul(complex_social_mat, complex_view_embeddings)
            norm_embeddings = tf.math.l2_normalize(complex_view_embeddings, axis=1)
            all_complex_embeddings += [norm_embeddings]

            # attribute view
            attribute_view_embeddings = tf.compat.v1.sparse_tensor_dense_matmul(attribute_mat,
                                                                                attribute_view_embeddings)
            norm_embeddings = tf.math.l2_normalize(attribute_view_embeddings, axis=1)
            all_attribute_embeddings += [norm_embeddings]
            # latent view
            latent_view_embeddings = tf.compat.v1.sparse_tensor_dense_matmul(latent_mat, latent_view_embeddings)
            norm_embeddings = tf.math.l2_normalize(latent_view_embeddings, axis=1)
            all_latent_embeddings += [norm_embeddings]

            # preference view
            ego_embeddings = tf.compat.v1.sparse_tensor_dense_matmul(ui_mat, ego_embeddings)
            norm_embeddings = tf.math.l2_normalize(ego_embeddings, axis=1)
            all_embeddings += [norm_embeddings]
            origin_embeddings_list += [ego_embeddings]
            # unlabeled sample view
            aug_social_embeddings = tf.compat.v1.sparse_tensor_dense_matmul(self.sub_social_mat['sub_social_mat'], aug_social_embeddings)
            norm_embeddings = tf.math.l2_normalize(aug_social_embeddings, axis=1)
            all_aug_social_embeddings += [norm_embeddings]

            aug_item_embeddings = tf.compat.v1.sparse_tensor_dense_matmul(self.sub_item_mat['sub_item_mat'], aug_item_embeddings)
            norm_embeddings = tf.math.l2_normalize(aug_item_embeddings, axis=1)
            all_aug_item_embeddings += [norm_embeddings]

        # averaging the view-specific embeddings
        # self.friend_view_embeddings = tf.reduce_sum(all_social_embeddings, axis=0)
        self.complex_view_embeddings = tf.reduce_sum(all_complex_embeddings, axis=0)
        self.sharing_view_embeddings = tf.reduce_sum(all_sharing_embeddings, axis=0)
        all_embeddings = tf.reduce_sum(all_embeddings, axis=0)
        self.rec_user_embeddings, self.rec_item_embeddings = tf.split(all_embeddings, [self.num_users, self.num_items],
                                                                      0)

        self.attribute_view_embeddings = tf.reduce_sum(all_attribute_embeddings, axis=0)
        self.latent_view_embeddings = tf.reduce_sum(all_latent_embeddings, axis=0)

        aug_social_embeddings = tf.reduce_sum(all_aug_social_embeddings, axis=0)
        self.aug_user_embeddings, self.no_use_item = tf.split(aug_social_embeddings, [self.num_users, self.num_items],
                                                                      0)   
        aug_item_embeddings = tf.reduce_sum(all_aug_item_embeddings, axis=0)
        self.no_use_item, self.aug_item_embeddings = tf.split(aug_item_embeddings, [self.num_users, self.num_items],
                                                                      0)    
        # embedding look-up
        self.batch_user_emb = tf.nn.embedding_lookup(self.rec_user_embeddings, self.u_idx)
        self.batch_pos_item_emb = tf.nn.embedding_lookup(self.rec_item_embeddings, self.v_idx)
        self.batch_neg_item_emb = tf.nn.embedding_lookup(self.rec_item_embeddings, self.neg_idx)

        self.center_embedding = origin_embeddings_list[0]
        self.context_embedding = origin_embeddings_list[self.hyper_layers * 2]

    def label_prediction(self, emb):
        emb = tf.nn.embedding_lookup(emb, tf.unique(self.u_idx)[0])
        emb = tf.nn.l2_normalize(emb, axis=1)
        aug_emb = tf.nn.embedding_lookup(self.aug_user_embeddings, tf.unique(self.u_idx)[0])
        aug_emb = tf.nn.l2_normalize(aug_emb, axis=1)
        prob = tf.matmul(emb, aug_emb, transpose_b=True)
        # avoid self-sampling
        # diag = tf.diag_part(prob)
        # prob = tf.matrix_diag(-diag)+prob
        prob = tf.nn.softmax(prob)
        return prob

    def label_item_prediction(self, emb):
        emb = tf.nn.embedding_lookup(emb, tf.unique(self.v_idx)[0])
        emb = tf.nn.l2_normalize(emb, axis=1)
        aug_emb = tf.nn.embedding_lookup(self.aug_item_embeddings, tf.unique(self.v_idx)[0])
        aug_emb = tf.nn.l2_normalize(aug_emb, axis=1)
        prob = tf.matmul(emb, aug_emb, transpose_b=True)
        # avoid self-sampling
        # diag = tf.diag_part(prob)
        # prob = tf.matrix_diag(-diag)+prob
        prob = tf.nn.softmax(prob)
        return prob

    # -ins_cnt = 10
    def sampling(self, logits):
        return tf.math.top_k(logits, self.instance_cnt)[1]

    def generate_pesudo_labels(self, prob1, prob2):
        positive = (prob1 + prob2) / 2
        pos_examples = self.sampling(positive)
        return pos_examples


    def cal_layer_loss(self, context_embedding, center_embedding, user, item):
        current_user_embeddings, current_item_embeddings = tf.split(context_embedding, [self.num_users, self.num_items],
                                                                    0)  # 1891*50, 15438*50
        # print(current_user_embeddings,':',current_item_embeddings)
        previous_user_embeddings_all, previous_item_embeddings_all = tf.split(center_embedding,
                                                                              [self.num_users, self.num_items],
                                                                              0)  # 1891*50, 15438*50
        # print(previous_user_embeddings_all,':',previous_item_embeddings_all)
        current_user_embeddings = tf.nn.embedding_lookup(current_user_embeddings, user)
        # print(current_user_embeddings,':')
        previous_user_embeddings = tf.nn.embedding_lookup(previous_user_embeddings_all, user)
        # print(previous_user_embeddings,':')

        norm_user_emb1 = tf.nn.l2_normalize(current_user_embeddings, axis=1)  #
        # print(norm_user_emb1,':')
        norm_user_emb2 = tf.nn.l2_normalize(previous_user_embeddings, axis=1)
        # print(norm_user_emb2,':')
        norm_all_user_emb = tf.nn.l2_normalize(previous_user_embeddings_all, axis=1)
        # print(norm_all_user_emb,':')
        # print(norm_all_user_emb,':')
        pos_score_user = tf.multiply(norm_user_emb1, norm_user_emb2)
        # print(pos_score_user,':')
        pos_score_user = tf.reduce_sum(pos_score_user, axis=1)
        # print(pos_score_user,':')
        norm_all_user_emb = tf.transpose(norm_all_user_emb, (1, 0))  #
        ttl_score_user = tf.matmul(norm_user_emb1, norm_all_user_emb)  # 错误提示：[2000,50] [1891,50]
        # print(ttl_score_user,':')
        pos_score_user = tf.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = tf.exp(ttl_score_user / self.ssl_temp)
        ttl_score_user = tf.reduce_sum(ttl_score_user, axis=1)
        # print(ttl_score_user,':')

        ssl_loss_user = -tf.compat.v1.log(pos_score_user / ttl_score_user)
        ssl_loss_user = tf.reduce_sum(ssl_loss_user)

        current_item_embeddings = tf.nn.embedding_lookup(current_item_embeddings, item)
        previous_item_embeddings = tf.nn.embedding_lookup(previous_item_embeddings_all, item)
        norm_item_emb1 = tf.nn.l2_normalize(current_item_embeddings, axis=1)
        norm_item_emb2 = tf.nn.l2_normalize(previous_item_embeddings, axis=1)
        norm_all_item_emb = tf.nn.l2_normalize(previous_item_embeddings_all, axis=1)

        pos_score_item = tf.multiply(norm_item_emb1, norm_item_emb2)
        pos_score_item = tf.reduce_sum(pos_score_item, axis=1)
        norm_all_item_emb = tf.transpose(norm_all_item_emb, (1, 0))
        ttl_score_item = tf.matmul(norm_item_emb1, norm_all_item_emb)
        pos_score_item = tf.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = tf.exp(ttl_score_item / self.ssl_temp)
        ttl_score_item = tf.reduce_sum(ttl_score_item, axis=1)

        ssl_loss_item = -tf.compat.v1.log(pos_score_item / ttl_score_item)
        ssl_loss_item = tf.reduce_sum(ssl_loss_item)

        ssl_layer_loss = ssl_loss_user + self.alpha * ssl_loss_item
        return ssl_layer_loss

    def neighbor_discrimination(self, positive, emb):
        def score(x1, x2):
            return tf.reduce_sum(tf.multiply(x1, x2), axis=2)

        emb = tf.nn.embedding_lookup(emb, tf.unique(self.u_idx)[0])
        emb = tf.nn.l2_normalize(emb, axis=1)
        aug_emb = tf.nn.embedding_lookup(self.aug_user_embeddings, tf.unique(self.u_idx)[0])
        aug_emb = tf.nn.l2_normalize(aug_emb, axis=1)
        pos_emb = tf.nn.embedding_lookup(aug_emb, positive)
        emb2 = tf.reshape(emb, [-1, 1, self.emb_size])
        emb2 = tf.tile(emb2, [1, self.instance_cnt, 1])
        pos = score(emb2, pos_emb)
        ttl_score = tf.matmul(emb, aug_emb, transpose_a=False, transpose_b=True)
        pos_score = tf.reduce_sum(tf.exp(pos / 0.1), axis=1)
        ttl_score = tf.reduce_sum(tf.exp(ttl_score / 0.1), axis=1)
        ssl_loss = -tf.reduce_sum(tf.compat.v1.log(pos_score / ttl_score))
        return ssl_loss

    def neighbor_item_discrimination(self, positive, emb):
        def score(x1, x2):
            return tf.reduce_sum(tf.multiply(x1, x2), axis=2)

        emb = tf.nn.embedding_lookup(emb, tf.unique(self.v_idx)[0])
        emb = tf.nn.l2_normalize(emb, axis=1)
        aug_emb = tf.nn.embedding_lookup(self.aug_item_embeddings, tf.unique(self.v_idx)[0])
        aug_emb = tf.nn.l2_normalize(aug_emb, axis=1)
        pos_emb = tf.nn.embedding_lookup(aug_emb, positive)
        emb2 = tf.reshape(emb, [-1, 1, self.emb_size])
        emb2 = tf.tile(emb2, [1, self.instance_cnt, 1])
        pos = score(emb2, pos_emb)
        ttl_score = tf.matmul(emb, aug_emb, transpose_a=False, transpose_b=True)
        pos_score = tf.reduce_sum(tf.exp(pos / 0.1), axis=1)
        ttl_score = tf.reduce_sum(tf.exp(ttl_score / 0.1), axis=1)
        ssl_loss = -tf.reduce_sum(tf.compat.v1.log(pos_score / ttl_score))
        return ssl_loss

    def trainModel(self):
        # training the recommendation model
        rec_loss = bpr_loss(self.batch_user_emb, self.batch_pos_item_emb, self.batch_neg_item_emb)
        rec_loss += self.regU * (tf.nn.l2_loss(self.user_embeddings) + tf.nn.l2_loss(self.item_embeddings))
        # self-supervision prediction
        # social_prediction = self.label_prediction(self.friend_view_embeddings)
        sharing_prediction = self.label_prediction(self.sharing_view_embeddings)
        rec_prediction = self.label_prediction(self.rec_user_embeddings)
        complex_prediction = self.label_prediction(self.complex_view_embeddings)

        attribute_prediction = self.label_item_prediction(self.attribute_view_embeddings)
        latent_prediction = self.label_item_prediction(self.latent_view_embeddings)
        rec_item_prediction = self.label_item_prediction(self.rec_item_embeddings)

        # find informative positive examples for each encoder
        self.f_pos = self.generate_pesudo_labels(sharing_prediction, rec_prediction)
        self.sh_pos = self.generate_pesudo_labels(complex_prediction, rec_prediction)
        self.r_pos = self.generate_pesudo_labels(complex_prediction, sharing_prediction)

        self.a_pos = self.generate_pesudo_labels(latent_prediction, rec_item_prediction)
        self.l_pos = self.generate_pesudo_labels(attribute_prediction, rec_item_prediction)
        self.rec_item_pos = self.generate_pesudo_labels(attribute_prediction, latent_prediction)

        # neighbor-discrimination based contrastive learning
        # user
        self.neighbor_dis_loss = self.neighbor_discrimination(self.f_pos, self.complex_view_embeddings)
        self.neighbor_dis_loss += self.neighbor_discrimination(self.sh_pos, self.sharing_view_embeddings)
        self.neighbor_dis_loss += self.neighbor_discrimination(self.r_pos, self.rec_user_embeddings)

        # item
        self.neighbor_dis_item_loss = self.neighbor_item_discrimination(self.a_pos, self.attribute_view_embeddings)
        self.neighbor_dis_item_loss += self.neighbor_item_discrimination(self.l_pos, self.latent_view_embeddings)
        self.neighbor_dis_item_loss += self.neighbor_item_discrimination(self.rec_item_pos, self.rec_item_embeddings)

        self.ssl_layer_loss = self.cal_layer_loss(self.context_embedding, self.center_embedding, self.u_idx, self.v_idx)

        # user = self.rating_mat.toarray()[self.u_idx]
        # pos_item = self.rating_mat.toarray()[self.v_idx]
        # neg_item = self.rating_mat.toarray()[self.neg_idx]

        # optimizer setting
        loss = rec_loss
        loss = loss + self.ss_rate * self.neighbor_dis_loss + self.ss_item_rate * self.neighbor_dis_item_loss
        loss = loss + self.struct_rate * self.ssl_layer_loss
        v1_opt = tf.compat.v1.train.AdamOptimizer(self.lRate)
        v1_op = v1_opt.minimize(rec_loss)
        v2_opt = tf.compat.v1.train.AdamOptimizer(self.lRate)
        v2_op = v2_opt.minimize(loss)
        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)
        for self.epoch in range(self.maxEpoch):
            # joint learning
            if self.epoch > self.maxEpoch / 3:
                sub_social_mat = {}
                sub_item_mat = {}
                sub_social_mat['adj_indices_sub'], sub_social_mat['adj_values_sub'], sub_social_mat[
                    'adj_shape_sub'] = self._convert_csr_to_sparse_tensor_inputs(
                    self.get_social_adj_mat(is_subgraph=True))
                sub_item_mat['adj_indices_sub'], sub_item_mat['adj_values_sub'], sub_item_mat[
                    'adj_shape_sub'] = self._convert_csr_to_sparse_tensor_inputs(
                    self.get_item_adj_mat(is_subgraph=True))
                for n, batch in enumerate(self.next_batch_pairwise()):
                    user_idx, i_idx, j_idx = batch
                    feed_dict = {self.u_idx: user_idx,
                                 self.v_idx: i_idx,
                                 self.neg_idx: j_idx}
                    feed_dict.update({
                        self.sub_social_mat['adj_values_sub']: sub_social_mat['adj_values_sub'],
                        self.sub_social_mat['adj_indices_sub']: sub_social_mat['adj_indices_sub'],
                        self.sub_social_mat['adj_shape_sub']: sub_social_mat['adj_shape_sub'],

                        self.sub_item_mat['adj_values_sub']: sub_item_mat['adj_values_sub'],
                        self.sub_item_mat['adj_indices_sub']: sub_item_mat['adj_indices_sub'],
                        self.sub_item_mat['adj_shape_sub']: sub_item_mat['adj_shape_sub'],

                    })
                    _, l1, l2, l3, l4 = self.sess.run(
                        [v2_op, rec_loss, self.neighbor_dis_loss, self.neighbor_dis_item_loss, self.ssl_layer_loss],
                        feed_dict=feed_dict)
                        # [v2_op, rec_loss, self.neighbor_dis_loss, self.neighbor_dis_item_loss, self.ssl_layer_loss],
                        # feed_dict=feed_dict)
                    print(self.foldInfo, 'training:', self.epoch + 1, 'batch', n, 'rec loss:', l1, 'con_loss:',
                          self.ss_rate * l2, 'item_loss:', self.ss_item_rate * l3, 'struct_loss:',
                          self.struct_rate * l4)
            else:
                # initialization with only recommendation task
                for n, batch in enumerate(self.next_batch_pairwise()):
                    user_idx, i_idx, j_idx = batch
                    feed_dict = {self.u_idx: user_idx,
                                 self.v_idx: i_idx,
                                 self.neg_idx: j_idx}
                    _, l1 = self.sess.run([v1_op, rec_loss],
                                          feed_dict=feed_dict)
                    print(self.foldInfo, 'training:', self.epoch + 1, 'batch', n, 'rec loss:', l1)
                print("当前是第"+self.epoch+"轮,时间是："+datetime.datetime.now())
            self.U, self.V = self.sess.run([self.rec_user_embeddings, self.rec_item_embeddings])
            self.ranking_performance(self.epoch)
        self.U, self.V = self.bestU, self.bestV

    def saveModel(self):
        self.bestU, self.bestV = self.sess.run([self.rec_user_embeddings, self.rec_item_embeddings])

    def predictForRanking(self, u):
        'invoked to rank all the items for the user'
        if self.data.containsUser(u):
            u = self.data.getUserId(u)
            return self.V.dot(self.U[u])
        else:
            return [self.data.globalMean] * self.num_items
