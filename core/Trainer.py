# -*- coding: utf-8 -*-

import csv
import logging
import math
import os
import random
import time

import cupy as cp
import numpy as np
from Model import Model
from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix
from scipy.sparse import lil_matrix as scipy_lil_matrix
from tqdm import tqdm


def calc_cos_similarity(a, b):
    return a.dot(b.transpose()) \
           / cp.linalg.norm(a, axis=1).reshape((a.shape[0], 1)) \
           / cp.linalg.norm(b, axis=1).reshape((1, b.shape[0]))


class Trainer:

    def __init__(self, model: Model):
        self.model = model
        self.report_loss_in_progress = False
        self.std_out = True
        self.out_loss_in_progress = False
        self.output_interval = 1
        self.orthogonal_vec_parameter = 250
        self.losses = []

        self.output_path = self.model.output_path
        os.makedirs(self.output_path, exist_ok=True)

        # Logger
        self.logger = logging.getLogger(type(self).__name__)
        self.handler = logging.FileHandler(filename=self.output_path + "/train.log")
        self.logger.addHandler(self.handler)

    def fit(self, include_be_used, is_test=False):
        print('training started')
        final_loss = 0.0
        if is_test:
            embeddings = self.model.embeddings_test = cp.vstack((self.model.embeddings_train, self.model.embeddings_test))
            relations_matrix = self.model.relations_matrix_test
        else:
            embeddings = self.model.embeddings_train
            relations_matrix = self.model.relations_matrix_train

        self.losses = [0.0 for _ in range(self.model.epochs)]
        start = time.time()
        if include_be_used:
            relations_matrix_transpose = relations_matrix.transpose().tocsr()
        else:
            relations_matrix_transpose = self.model.relations_matrix_test_not_be_used_from_other_class.transpose().tocsr()
        self.model.loop_counter_for_negative_sampling = 0

        for epoch in tqdm(range(self.model.epochs)):

            learn_rate = self.calc_learn_rate(epoch)
            target_embeddings = relations_matrix.dot(embeddings)

            if epoch+1 == self.model.epochs:
                if is_test:
                    last_loss = self.calc_loss \
                        (embeddings[self.model.num_of_train_nodes:],
                         target_embeddings[self.model.num_of_train_nodes:])
                else:
                    last_loss = self.calc_loss(embeddings, target_embeddings)

                if math.isnan(last_loss):
                    return 999999
                self.losses[epoch] = last_loss
                final_loss = last_loss

            # 各要素の想定ベクトルを計算してそれを用いて算出した傾きを元に勾配降下
            self.calc_grad_and_update_embeddings(embeddings, target_embeddings, is_test, learn_rate, relations_matrix_transpose)
            # negative sampling
            # if do_negative_sampling:
            self.negative_sampling(embeddings, is_test)
            # norm罰則計算
            self.norm_penalty(embeddings, is_test, learn_rate)

        print('elapsed_time for training : {}'.format(time.time() - start))
        return final_loss

    @staticmethod
    def calc_loss(embeddings, target_embeddings):
        norm = cp.linalg.norm(embeddings, axis=1).reshape((embeddings.shape[0], 1))
        norm_loss = float(cp.sum(cp.square(1 - norm) / norm))
        target_loss = float(cp.sum(cp.square(embeddings - target_embeddings) / 2))
        loss = norm_loss + target_loss
        return loss

    def calc_grad_and_update_embeddings(self, embeddings, target_embeddings, is_test, learn_rate, relations_matrix_transpose):
        # 着目ノードのベクトルと，その内部情報を元に算出した着目ノードの想定ベクトルとの差
        original_minus_target = embeddings - target_embeddings
        # 着目ノードを継承クラスや呼び出しメソッドとして内部で使っているノードのベクトルと，
        # そのノードの想定ベクトルとの差の和
        original_minus_target_be_used = - relations_matrix_transpose.dot(original_minus_target)
        original_minus_target_all = original_minus_target + original_minus_target_be_used
        norm_original_minus_target_all = cp.linalg.norm(original_minus_target_all, axis=1).reshape((original_minus_target_all.shape[0], 1))

        if is_test:
            embeddings[self.model.num_of_train_nodes:] \
                -= learn_rate * cp.nan_to_num(original_minus_target_all / norm_original_minus_target_all)[self.model.num_of_train_nodes:]
        else:
            embeddings -= learn_rate * cp.nan_to_num(original_minus_target_all / norm_original_minus_target_all)

    def negative_sampling(self, embeddings, is_test):
        if is_test:
            prepare_func = self.prepare_test_negative_sampling()
        else:
            prepare_func = self.prepare_train_negative_sampling()
        for _ in range(self.model.negative_sample_num):
            negative_sampling_matrix = prepare_func
            negative_embeddings = negative_sampling_matrix.dot(embeddings)
            norm_embeddings = cp.linalg.norm(embeddings, axis=1).reshape((embeddings.shape[0], 1))

            target_negative_cos = cp.cos(embeddings, negative_embeddings)
            new_embeddings = embeddings + (embeddings - negative_embeddings) \
                             * self.model.target_negative_weight * cp.cos(cp.arccos(target_negative_cos) / 2)
            norm_new_embeddings = cp.linalg.norm(new_embeddings, axis=1).reshape((new_embeddings.shape[0], 1))
            embeddings = new_embeddings / norm_new_embeddings * norm_embeddings

    def prepare_test_negative_sampling(self):
        row_size = col_size = self.model.num_of_all_nodes
        target_size = self.model.num_of_test_nodes
        not_candidates_index_dict = self.model.not_negative_sampling_candidates_index_dict_test

        negative_sampling_matrix = scipy_lil_matrix((row_size, col_size), dtype=np.float32)

        for target_row_index in range(target_size):
            target_row_index = self.model.num_of_train_nodes + target_row_index
            while True:
                negative_sample_index = random.randrange(0, row_size)
                if negative_sample_index in not_candidates_index_dict[target_row_index]:
                    continue
                else:
                    break
            negative_sampling_matrix[target_row_index, negative_sample_index] = 1

        return cupy_csr_matrix(negative_sampling_matrix.tocsr())

    def prepare_train_negative_sampling(self):
        row_size = col_size = target_size = self.model.num_of_train_nodes
        not_candidates_index_dict = self.model.not_negative_sampling_candidates_index_dict_train

        negative_sampling_matrix = scipy_lil_matrix((row_size, col_size), dtype=np.float32)

        for target_row_index in range(target_size):
            while True:
                negative_sample_index = random.randrange(0, row_size)
                if negative_sample_index in not_candidates_index_dict[target_row_index]:
                    continue
                else:
                    break
            negative_sampling_matrix[target_row_index, negative_sample_index] = 1

        return cupy_csr_matrix(negative_sampling_matrix.tocsr())

    def norm_penalty(self, embeddings, is_test, learn_rate):
        if is_test:
            tmp_embeddings = embeddings[self.model.num_of_train_nodes:]
        else:
            tmp_embeddings = embeddings
        norm = cp.linalg.norm(tmp_embeddings, axis=1).reshape((tmp_embeddings.shape[0], 1))
        squared_norm = cp.square(norm)
        norm_gradient = (- cp.reciprocal(squared_norm) + 1) * tmp_embeddings / norm
        tmp_embeddings -= learn_rate * norm_gradient

    def calc_learn_rate(self, epoch: int) -> float:
        if self.model.decay == 'reciprocal':
            return self.model.initial_learn_rate / epoch
        elif self.model.decay == 'exp9':
            return self.model.initial_learn_rate * (math.pow(0.9, 100 * epoch / self.model.epochs))
        elif self.model.decay == 'exp99':
            return self.model.initial_learn_rate * (math.pow(0.99, 100 * epoch / self.model.epochs))
        elif self.model.decay == 'exp999':
            return self.model.initial_learn_rate * (math.pow(0.999, 100 * epoch / self.model.epochs))
        elif self.model.decay == 'linear':
            return self.model.initial_learn_rate * (1 - (epoch / (self.model.epochs + 1)))
        elif self.model.decay == 'gentledec':
            return self.model.initial_learn_rate * (100 / (100 + (100 * epoch / self.model.epochs)))
        else:
            raise ValueError

    def save_train_log_as_csv(self, is_test):
        """
        訓練中の各ループにおける学習率・勾配・損失の値をCSV形式でファイルに出力する
        """
        if is_test:
            trainORtest = 'test'
        else:
            trainORtest = 'train'

        path = '{}/train_log/'.format(self.model.output_path)
        with open('{}{}-{}.csv'.format(path, trainORtest, self.model.model_name), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss'])
            for epoch_1, loss in enumerate(self.losses):
                writer.writerow([epoch_1+1, loss])
