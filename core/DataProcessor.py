# -*- coding:utf-8 -*-

import codecs
import copy
import json
import os
import sys
import time
from collections import defaultdict

import cupy as cp
import numpy as np
from cupyx.scipy.sparse import csr_matrix as cupy_csr_matrix
from scipy.sparse import lil_matrix as scipy_lil_matrix
from tqdm import tqdm

from . import Model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_HOME = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))

sys.path.append(os.path.join(SCRIPT_DIR, os.pardir))


class DataProcessor:
    def __init__(self, model: Model):
        self.model = model
        self.model.loop_counter_for_negative_sampling \
            = self.model.num_of_train_nodes_and_recommend_class = self.model.num_of_test_classes \
            = self.model.num_of_train_classes = self.model.num_of_classes = self.model.num_of_methods \
            = self.model.num_of_fields = self.model.num_of_all_nodes = self.model.num_of_all_edges \
            = self.model.num_of_train_nodes = self.model.num_of_test_nodes = 0
        self.model.how_many_used_dict = defaultdict(int)

    def pre_process(self):
        """
        データセットを読み込んで，要素と分散表現行列のインデックス対応付けと，関係をモデリングした行列の生成
        :return:
        """
        print('data process started')
        start = time.time()
        self.json_parse()
        self.duplicate_delete()
        self.split_all_elements_into_set()
        self.set_and_count_nodes()
        self.calc_num_of_train_and_recommend_nodes()
        self.name_indexing()
        self.matrix_initialize()
        self.set_relations()
        self.lil_matrix_to_csr()
        self.count_edges()
        print('data process finished')
        print('elapsed_time for data process : {}'.format(time.time() - start))
        return self.model

    def set_relations(self):
        self.set_class_relations(is_test=False)
        self.set_method_relations(is_test=False)
        self.set_field_relations(is_test=False)
        self.set_class_relations(is_test=True)
        self.set_method_relations(is_test=True)
        self.set_field_relations(is_test=True)

    def count_edges(self):
        self.model.num_of_class_extends_relations = sum([len(x) for x in self.model.class_extends_relations.values()])
        self.model.num_of_method_in_class_relations = sum([len(x) for x in self.model.method_in_class_relations.values()])
        self.model.num_of_field_in_class_relations = sum([len(x) for x in self.model.field_in_class_relations.values()])
        self.model.num_of_method_call_relations = sum([len(x) for x in self.model.method_call_relations.values()])
        self.model.num_of_field_in_method_relations = sum([len(x) for x in self.model.field_in_method_relations.values()])
        self.model.num_of_return_type_relations = sum([len(x) for x in self.model.return_type_relations.values()])
        self.model.num_of_field_type_relations = sum([len(x) for x in self.model.field_type_relations.values()])

        # 全エッジのカウント
        self.model.num_of_all_edges = self.model.num_of_class_extends_relations + \
                                      self.model.num_of_method_in_class_relations + \
                                      self.model.num_of_field_in_class_relations + \
                                      self.model.num_of_method_call_relations + \
                                      self.model.num_of_field_in_method_relations + \
                                      self.model.num_of_return_type_relations + \
                                      self.model.num_of_field_type_relations
        # print("全ノード数 : {}, 全エッジ数 : {}".format(self.model.num_of_all_nodes, self.model.num_of_all_edges))

    def duplicate_delete(self):
        class_set = copy.deepcopy(self.model.classes)
        method_set = copy.deepcopy(self.model.methods)
        field_set = copy.deepcopy(self.model.fields)

        self.model.declaration_classes_all = self.model.declaration_classes_all - method_set - field_set
        self.model.methods_in_class_all = self.model.methods_in_class_all - field_set - class_set
        self.model.fields_in_class_all = self.model.fields_in_class_all - class_set - method_set

        self.model.classes = self.model.classes - method_set - field_set
        self.model.methods = self.model.methods - field_set - class_set
        self.model.fields = self.model.fields - class_set - method_set

    def set_and_count_nodes(self):
        self.model.all_nodes = self.model.classes | self.model.methods | self.model.fields
        self.model.all_test_nodes = self.model.classes_test | self.model.methods_test | self.model.fields_test
        self.model.all_train_nodes = self.model.classes_train | self.model.methods_train | self.model.fields_train
        self.model.num_of_train_classes = len(self.model.classes_train)
        self.model.num_of_all_nodes = len(self.model.all_nodes)
        self.model.num_of_test_nodes = len(self.model.all_test_nodes)
        self.model.num_of_train_nodes = len(self.model.all_train_nodes)

        self.model.num_of_test_classes = len(self.model.classes_test)
        self.model.num_of_train_nodes_and_test_class = self.model.num_of_train_nodes + self.model.num_of_test_classes

    def lil_matrix_to_csr(self):
        self.model.relations_matrix_train = cupy_csr_matrix(self.model.relations_matrix_train.tocsr(), dtype=cp.float32)
        self.model.relations_matrix_test = cupy_csr_matrix(self.model.relations_matrix_test.tocsr(), dtype=cp.float32)
        self.model.relations_matrix_test_not_be_used_from_other_class \
            = cupy_csr_matrix(self.model.relations_matrix_test_not_be_used_from_other_class.tocsr(), dtype=cp.float32)

    def json_parse(self):
        """
        jsonファイル群をパースして，各関係を保持する辞書に値を入れる
        :return:
        """
        # train_test_dirsはtrain, testの2つ
        train_test_dirs = os.listdir(self.model.input_path)
        for train_or_test_dir in tqdm(train_test_dirs):
            dataset_path = os.path.join(self.model.input_path, train_or_test_dir)
            if not os.path.isdir(dataset_path):
                continue

            if train_or_test_dir == 'test':
                is_test = True
            else:
                is_test = False

            relations = os.listdir(dataset_path)
            for relation in relations:
                relation_path = os.path.join(dataset_path, relation)
                if not os.path.isdir(relation_path):
                    continue

                file_list = os.listdir(relation_path)
                for json_file_name in file_list:
                    json_file_path = os.path.join(relation_path, json_file_name)
                    if not os.path.isfile(json_file_path):
                        continue
                    with open(json_file_path, 'r', encoding='utf-8') as f:
                        single_json = json.load(f)
                        self.json_to_relation(single_json, relation, is_test)

    def calc_num_of_train_and_recommend_nodes(self):
        """
        テストクラス，メソッド，フィールドを計算し，その後テストノード合計値も計算する
        :return:
        """
        self.model.num_of_train_nodes_and_recommend_class \
            = self.model.num_of_train_nodes + len(self.model.declaration_classes_test)

    def name_indexing(self):
        """
        識別子名とembeddingsにおけるインデックスの対応付け処理
        :return:
        """
        count = 0
        program_elements = [self.model.declaration_classes_train, self.model.not_declaration_classes_train,
                            self.model.methods_in_class_train, self.model.not_methods_in_class_train,
                            self.model.fields_in_class_train, self.model.not_fields_in_class_train,
                            self.model.declaration_classes_test, self.model.not_declaration_classes_test,
                            self.model.methods_in_class_test, self.model.not_methods_in_class_test,
                            self.model.fields_in_class_test, self.model.not_fields_in_class_test]

        for program_element in tqdm(program_elements):
            for e in copy.deepcopy(program_element):
                self.model.name_to_index[e] = count
                self.model.index_to_name[count] = e
                count += 1

    def matrix_initialize(self):
        """
        relations_matrixと，embeddingsとnegative_sampling_not_candidates_matrixの初期化．
        :return:
        """
        cp.random.seed(0)
        # 訓練ノードの学習には訓練ノードのみを用いるので，行数列数ともに訓練ノード数
        self.model.relations_matrix_train \
            = scipy_lil_matrix((self.model.num_of_train_nodes, self.model.num_of_train_nodes), dtype=np.float32)

        self.model.embeddings_train = cp.random.rand(self.model.num_of_train_nodes, self.model.embedding_dim) \
                                          .astype(cp.float32) * 2 - 1
        self.model.embeddings_train = self.model.embeddings_train \
                                      / cp.linalg.norm(self.model.embeddings_train, axis=1) \
                                          .reshape((self.model.embeddings_train.shape[0], 1))

        self.model.relations_matrix_test \
            = scipy_lil_matrix((self.model.num_of_all_nodes, self.model.num_of_all_nodes), dtype=np.float32)
        self.model.relations_matrix_test_not_be_used_from_other_class \
            = scipy_lil_matrix((self.model.num_of_all_nodes, self.model.num_of_all_nodes), dtype=np.float32)

        self.model.embeddings_test = cp.random.rand(self.model.num_of_test_nodes, self.model.embedding_dim) \
                                         .astype(cp.float32) * 2 - 1
        self.model.embeddings_test = self.model.embeddings_test \
                                     / cp.linalg.norm(self.model.embeddings_test, axis=1) \
                                         .reshape((self.model.embeddings_test.shape[0], 1))
        self.model.not_negative_sampling_candidates_index_dict_test = defaultdict(set)
        self.model.not_negative_sampling_candidates_index_dict_train = defaultdict(set)

    def set_class_relations(self, is_test=False):
        """
        着目クラスと継承クラス，所有メソッド群，所有フィールド群との関係を行列に反映する処理．
        それぞれ対応する要素がないときは自分自身のインデックスに値を入れることで他要素との比率が崩れるのを防ぐ
        関係があるノードと自分自身はネガティブサンプリングの対象から外す．
        着目クラスと他の両方がテストノードだったらそれは訓練でもテストでも使わない．
        推薦対象のクラス（テストノードに含まれているクラス）に関する行列も作っておく．
        :return:
        """
        if is_test:
            classes = self.model.declaration_classes_all
        else:
            classes = self.model.declaration_classes_train

        for i, class_element in tqdm(enumerate(classes)):
            have_parent = class_element in self.model.class_extends_relations
            have_methods = class_element in self.model.method_in_class_relations
            have_fields = class_element in self.model.field_in_class_relations

            # 訓練fit時には，テストノードは使えない
            parent = methods_in_class = methods_in_class_num = fields_in_class = fields_in_class_num = None
            if have_parent:
                parent = self.model.class_extends_relations[class_element]
                if not is_test and parent in self.model.classes_test:
                    have_parent = False

            if have_methods:
                methods_in_class = self.model.method_in_class_relations[class_element]
                if not is_test:
                    methods_in_class = methods_in_class & self.model.methods_train
                methods_in_class_num = len(methods_in_class)
                if methods_in_class_num == 0:
                    have_methods = False

            if have_fields:
                fields_in_class = self.model.field_in_class_relations[class_element]
                if not is_test:
                    fields_in_class = fields_in_class & self.model.fields_train
                fields_in_class_num = len(fields_in_class)
                if fields_in_class_num == 0:
                    have_fields = False

            class_weight = self.model.alpha
            method_weight = self.model.beta
            field_weight = 1. - self.model.alpha - self.model.beta

            # 継承クラス，所有メソッド群，所有フィールド群がそれぞれあるかどうかによって重みを動的に変更する
            if have_parent:
                if have_methods:
                    if have_fields:
                        pass
                    else:
                        class_weight += field_weight / 2
                        method_weight += field_weight / 2
                        field_weight = 0
                else:
                    if have_fields:
                        class_weight += method_weight / 2
                        field_weight += method_weight / 2
                        method_weight = 0
                    else:
                        class_weight = 1
                        method_weight = field_weight = 0
            else:
                if have_methods:
                    if have_fields:
                        method_weight += class_weight / 2
                        field_weight += class_weight / 2
                        class_weight = 0
                    else:
                        method_weight = 1
                        class_weight = field_weight = 0
                else:
                    if have_fields:
                        field_weight = 1
                        class_weight = method_weight = 0
                    else:
                        continue

            if is_test:
                relations_matrix = self.model.relations_matrix_test
                not_negative_sampling_candidates_index_dict = self.model.not_negative_sampling_candidates_index_dict_test
            else:
                relations_matrix = self.model.relations_matrix_train
                not_negative_sampling_candidates_index_dict = self.model.not_negative_sampling_candidates_index_dict_train

            class_index = self.model.name_to_index[class_element]
            not_negative_sampling_candidates_index_dict[class_index].add(class_index)

            # 着目クラスと継承クラスとの関係の処理．
            if have_parent:
                parent_index = self.model.name_to_index[parent]
                relations_matrix[class_index, parent_index] = class_weight
                if is_test:
                    self.model.relations_matrix_test_not_be_used_from_other_class[class_index, parent_index] = class_weight
                not_negative_sampling_candidates_index_dict[class_index].add(parent_index)
                not_negative_sampling_candidates_index_dict[parent_index].add(class_index)
                self.model.how_many_used_dict[class_element] += 1
            else:
                relations_matrix[class_index, class_index] += class_weight

            # 着目クラスと所有メソッド群との関係の処理．
            if have_methods:
                method_in_class_weight = method_weight / methods_in_class_num
                for method_in_class in methods_in_class:
                    method_in_class_index = self.model.name_to_index[method_in_class]
                    relations_matrix[class_index, method_in_class_index] = method_in_class_weight
                    if is_test:
                        self.model.relations_matrix_test_not_be_used_from_other_class[class_index, method_in_class_index] = method_in_class_weight
                    not_negative_sampling_candidates_index_dict[class_index].add(method_in_class_index)
                    not_negative_sampling_candidates_index_dict[method_in_class_index].add(class_index)
                    self.model.how_many_used_dict[class_element] += 1
            else:
                relations_matrix[class_index, class_index] += method_weight

            # 着目クラスと所有フィールド群との関係の処理
            if have_fields:
                field_in_class_weight = field_weight / fields_in_class_num
                for field_in_class in fields_in_class:
                    field_in_class_index = self.model.name_to_index[field_in_class]
                    relations_matrix[class_index, field_in_class_index] = field_in_class_weight
                    if is_test:
                        self.model.relations_matrix_test_not_be_used_from_other_class[class_index, field_in_class_index] = field_in_class_weight
                    not_negative_sampling_candidates_index_dict[class_index].add(field_in_class_index)
                    not_negative_sampling_candidates_index_dict[field_in_class_index].add(class_index)
                    self.model.how_many_used_dict[class_element] += 1
            else:
                relations_matrix[class_index, class_index] += field_weight

    def set_method_relations(self, is_test=False):
        """
        着目メソッドと呼び出しメソッド群，アクセスしているフィールド群，戻り値のクラスとの関係を行列に反映する処理．
        それぞれ対応する要素がないときは自分自身のインデックスに値を入れることで他要素との比率が崩れるのを防ぐ
        関係があるノードと自分自身はネガティブサンプリングの対象から外す．
        着目メソッドと他の両方がテストノードだったらそれは訓練でもテストでも使わない．
        :return:
        """
        if is_test:
            methods = self.model.methods_in_class_all
        else:
            methods = self.model.methods_in_class_train

        for i, method_element in tqdm(enumerate(methods)):
            do_call = method_element in self.model.method_call_relations
            do_access = method_element in self.model.field_in_method_relations
            do_return = method_element in self.model.return_type_relations

            # 訓練fit時には，テストノードは使えない
            callee_methods = callee_methods_num = fields_in_method = fields_in_method_num = return_type = None
            if do_call:
                callee_methods = self.model.method_call_relations[method_element]
                if not is_test:
                    callee_methods = callee_methods & self.model.methods_train
                callee_methods_num = len(callee_methods)

                if callee_methods_num == 0:
                    do_call = False

            if do_access:
                fields_in_method = self.model.field_in_method_relations[method_element]
                if not is_test:
                    fields_in_method = fields_in_method & self.model.fields_train
                fields_in_method_num = len(fields_in_method)

                if fields_in_method_num == 0:
                    do_access = False

            if do_return:
                return_type = self.model.return_type_relations[method_element]
                if not is_test and return_type in self.model.classes_test:
                    do_return = False

            callee_weight = self.model.gamma
            access_weight = self.model.delta
            return_weight = 1. - self.model.gamma - self.model.delta

            # 継承クラス，所有メソッド群，所有フィールド群がそれぞれあるかどうかによって重みを動的に変更する
            if do_call:
                if do_access:
                    if do_return:
                        pass
                    else:
                        callee_weight += return_weight / 2
                        access_weight += return_weight / 2
                        return_weight = 0
                else:
                    if do_return:
                        callee_weight += access_weight / 2
                        return_weight += access_weight / 2
                        access_weight = 0
                    else:
                        pass
            else:
                if do_access:
                    if do_return:
                        access_weight += callee_weight / 2
                        return_weight += callee_weight / 2
                        callee_weight = 0
                    else:
                        access_weight = 1
                        callee_weight = return_weight = 0
                else:
                    if do_return:
                        return_weight = 1
                        callee_weight = access_weight = 0
                    else:
                        continue

            if is_test:
                relations_matrix = self.model.relations_matrix_test
                not_negative_sampling_candidates_index_dict = self.model.not_negative_sampling_candidates_index_dict_test
            else:
                relations_matrix = self.model.relations_matrix_train
                not_negative_sampling_candidates_index_dict = self.model.not_negative_sampling_candidates_index_dict_train

            method_index = self.model.name_to_index[method_element]
            not_negative_sampling_candidates_index_dict[method_index].add(method_index)

            # 宣言メソッドと，呼び出しメソッド群やアクセスフィールド群や戻り値の型のクラスが，同一クラスのものか判定する下準備
            method_parents = set()
            method_parent = self.model.method_in_class_relations_reverse[method_element]
            method_parents.add(method_parent)
            method_parents.add(self.model.class_extends_relations[method_parent])

            # 呼び出し関係の処理
            if do_call:
                call_weight = callee_weight / callee_methods_num
                for callee in callee_methods:
                    callee_index = self.model.name_to_index[callee]
                    relations_matrix[method_index, callee_index] = call_weight
                    not_negative_sampling_candidates_index_dict[method_index].add(callee_index)
                    not_negative_sampling_candidates_index_dict[callee_index].add(method_index)

                    if not is_test:
                        continue
                    callee_parents = set()
                    callee_parent = self.model.method_in_class_relations_reverse[callee]
                    callee_parents.add(callee_parent)
                    callee_parents.add(self.model.class_extends_relations[callee_parent])
                    if method_parents & callee_parents:
                        self.model.relations_matrix_test_not_be_used_from_other_class[method_index, callee_index] = call_weight
            else:
                relations_matrix[method_index, method_index] += callee_weight

            # フィールドにアクセスしている関係の処理
            if do_access:
                field_access_weight = access_weight / fields_in_method_num
                for field_in_method in fields_in_method:
                    field_in_method_index = self.model.name_to_index[field_in_method]
                    relations_matrix[method_index, field_in_method_index] = field_access_weight
                    not_negative_sampling_candidates_index_dict[method_index].add(field_in_method_index)
                    not_negative_sampling_candidates_index_dict[field_in_method_index].add(method_index)

                    if not is_test:
                        continue
                    field_parents = set()
                    field_parent = self.model.field_in_class_relations_reverse[field_in_method]
                    field_parents.add(field_parent)
                    field_parents.add(self.model.class_extends_relations[field_parent])
                    if method_parents & field_parents:
                        self.model.relations_matrix_test_not_be_used_from_other_class[method_index, field_in_method_index] = field_access_weight
            else:
                relations_matrix[method_index, method_index] = access_weight

            # 戻り値のクラスの関係の処理
            if do_return:
                return_type_index = self.model.name_to_index[return_type]
                relations_matrix[method_index, return_type_index] = return_weight
                not_negative_sampling_candidates_index_dict[method_index].add(return_type_index)
                not_negative_sampling_candidates_index_dict[return_type_index].add(method_index)
                if is_test and return_type in method_parents:
                    self.model.relations_matrix_test_not_be_used_from_other_class[method_index, return_type_index] = return_weight
            else:
                relations_matrix[method_index, method_index] += return_weight

    def set_field_relations(self, is_test=False):
        """
        着目フィールドとの型との関係を行列に反映する処理．
        それぞれ対応する要素がないときは自分自身のインデックスに値を入れることで他要素との比率が崩れるのを防ぐ
        関係があるノードと自分自身はネガティブサンプリングの対象から外す．
        着目フィールドと他の両方がテストノードだったらそれは訓練でもテストでも使わない．
        :return:
        """

        if is_test:
            fields = self.model.fields_in_class_all
        else:
            fields = self.model.fields_in_class_train

        for i, field_element in tqdm(enumerate(fields)):
            have_type = field_element in self.model.field_type_relations
            field_type = None
            if have_type:
                field_type = self.model.field_type_relations[field_element]
                if not is_test and field_type in self.model.classes_test:
                    have_type = False

            if is_test:
                relations_matrix = self.model.relations_matrix_test
                not_negative_sampling_candidates_index_dict = self.model.not_negative_sampling_candidates_index_dict_test
            else:
                relations_matrix = self.model.relations_matrix_train
                not_negative_sampling_candidates_index_dict = self.model.not_negative_sampling_candidates_index_dict_train

            field_index = self.model.name_to_index[field_element]
            not_negative_sampling_candidates_index_dict[field_index].add(field_index)

            # フィールドと型との関係の処理
            if have_type:
                field_type_index = self.model.name_to_index[field_type]
                relations_matrix[field_index, field_type_index] = 1.
                not_negative_sampling_candidates_index_dict[field_index].add(field_type_index)
                not_negative_sampling_candidates_index_dict[field_type_index].add(field_index)

                if not is_test:
                    continue

                field_parents = set()
                field_parent = self.model.field_in_class_relations_reverse[field_element]
                field_parents.add(field_parent)
                field_parents.add(self.model.class_extends_relations[field_parent])
                if field_type in field_parents:
                    self.model.relations_matrix_test_not_be_used_from_other_class[field_index, field_type_index] = 1.
            else:
                continue
        # print('終わり')

    def set_log(self):
        log_dir = self.model.log_dir
        write_json(log_dir, "callee_methods.json", self.model.callee_methods)

    def json_to_relation(self, single_json, relation: str, is_test: bool):
        if is_test:
            classes_train_or_test = self.model.classes_test
            methods_train_or_test = self.model.methods_test
            fields_train_or_test = self.model.fields_test
        else:
            classes_train_or_test = self.model.classes_train
            methods_train_or_test = self.model.methods_train
            fields_train_or_test = self.model.fields_train

        for key, values in single_json.items():
            key_name = key.split('<')[0].split('[')[0].strip('\'{}(),')

            value_names = []
            for value in values:
                value_name = value.split('<')[0].split('[')[0].strip('\'{}(),')
                value_names.append(value_name)

            if relation == 'classExtends':
                classes_train_or_test.add(key_name)
                self.model.classes.add(key_name)
                self.model.declaration_classes_all.add(key_name)

                value_name = value_names[0]
                classes_train_or_test.add(value_name)
                self.model.classes.add(value_name)
                self.model.super_classes.add(value_name)

                self.model.class_extends_relations[key_name] = value_name
                self.model.class_extends_relations_reverse[value_name].add(key_name)

            elif relation == 'methodInClass':
                classes_train_or_test.add(key_name)
                self.model.classes.add(key_name)
                self.model.declaration_classes_all.add(key_name)

                for value_name in value_names:
                    methods_train_or_test.add(value_name)
                    self.model.methods.add(value_name)
                    self.model.methods_in_class_all.add(value_name)

                    self.model.method_in_class_relations[key_name].add(value_name)
                    self.model.method_in_class_relations_reverse[value_name] = key_name

            elif relation == 'fieldInClass':
                classes_train_or_test.add(key_name)
                self.model.classes.add(key_name)
                self.model.declaration_classes_all.add(key_name)

                for value_name in value_names:
                    fields_train_or_test.add(value_name)
                    self.model.fields.add(value_name)
                    self.model.fields_in_class_all.add(value_name)

                    self.model.field_in_class_relations[key_name].add(value_name)
                    self.model.field_in_class_relations_reverse[value_name] = key_name

            elif relation == 'methodCall':
                methods_train_or_test.add(key_name)
                self.model.methods.add(key_name)
                self.model.methods_in_class_all.add(key_name)

                for value_name in value_names:
                    methods_train_or_test.add(value_name)
                    self.model.methods.add(value_name)
                    self.model.callee_methods.add(value_name)

                    self.model.method_call_relations[key_name].add(value_name)
                    self.model.method_call_relations_reverse[value_name].add(key_name)

            elif relation == 'fieldInMethod':
                methods_train_or_test.add(key_name)
                self.model.methods.add(key_name)
                self.model.methods_in_class_all.add(key_name)

                for value_name in value_names:
                    fields_train_or_test.add(value_name)
                    self.model.fields.add(value_name)
                    self.model.fields_in_method.add(value_name)

                    self.model.field_in_method_relations[key_name].add(value_name)
                    self.model.field_in_method_relations_reverse[value_name].add(key_name)

            elif relation == 'returnType':
                methods_train_or_test.add(key_name)
                self.model.methods.add(key_name)
                self.model.methods_in_class_all.add(key_name)

                value_name = value_names[0]

                classes_train_or_test.add(value_name)
                self.model.classes.add(value_name)
                self.model.return_type_classes.add(value_name)

                self.model.return_type_relations[key_name] = value_name
                self.model.return_type_relations_reverse[value_name].add(key_name)

            elif relation == 'fieldType':
                fields_train_or_test.add(key_name)
                self.model.fields.add(key_name)
                self.model.fields_in_class_all.add(key_name)

                value_name = value_names[0]
                classes_train_or_test.add(value_name)
                self.model.classes.add(value_name)
                self.model.field_type_classes.add(value_name)

                self.model.field_type_relations[key_name] = value_name
                self.model.field_type_relations_reverse[value_name].add(key_name)

    def split_all_elements_into_set(self):
        """
        各要素を訓練orテスト，宣言クラスメソッドフィールドかどうかなどで正しく振り分ける
        :return:
        """
        self.model.not_declaration_classes_all = self.model.classes - self.model.declaration_classes_all
        self.model.not_methods_in_class_all = self.model.methods - self.model.methods_in_class_all
        self.model.not_fields_in_class_all = self.model.fields - self.model.fields_in_class_all

        self.model.declaration_classes_test = self.model.declaration_classes_all & self.model.classes_test
        self.model.declaration_classes_train = self.model.declaration_classes_all - self.model.declaration_classes_test
        self.model.not_declaration_classes_test = self.model.not_declaration_classes_all & self.model.classes_test
        self.model.not_declaration_classes_train = self.model.not_declaration_classes_all - self.model.not_declaration_classes_test

        self.model.methods_in_class_test = self.model.methods_in_class_all & self.model.methods_test
        self.model.methods_in_class_train = self.model.methods_in_class_all - self.model.methods_in_class_test
        self.model.not_methods_in_class_test = self.model.not_methods_in_class_all & self.model.methods_test
        self.model.not_methods_in_class_train = self.model.not_methods_in_class_all - self.model.not_methods_in_class_test

        self.model.fields_in_class_test = self.model.fields_in_class_all & self.model.fields_test
        self.model.fields_in_class_train = self.model.fields_in_class_all - self.model.fields_in_class_test
        self.model.not_fields_in_class_test = self.model.not_fields_in_class_all & self.model.fields_test
        self.model.not_fields_in_class_train = self.model.not_fields_in_class_all - self.model.not_fields_in_class_test

        self.integrate_set()

    def integrate_set(self):
        self.model.classes_test = self.model.declaration_classes_test | self.model.not_declaration_classes_test
        self.model.classes_train = self.model.declaration_classes_train | self.model.not_declaration_classes_train
        self.model.methods_test = self.model.methods_in_class_test | self.model.not_methods_in_class_test
        self.model.methods_train = self.model.methods_in_class_train | self.model.not_methods_in_class_train
        self.model.fields_test = self.model.fields_in_class_test | self.model.not_fields_in_class_test
        self.model.fields_train = self.model.fields_in_class_train | self.model.not_fields_in_class_train

        self.model.classes = self.model.classes_test | self.model.classes_train
        self.model.methods = self.model.methods_test | self.model.methods_train
        self.model.fields = self.model.fields_test | self.model.fields_train


def write_json(out_dir, filename, data):
    """
    データをjson形式でファイルに出力する
    :param filename: 出力するファイルの名前
    :param data: 出力するデータ
    """
    # todo 受け付けるデータ形式が明確でない(受け入れない形式だとエラーを吐く)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    dump_file = codecs.open(os.path.join(out_dir, filename), "w+")
    json.dump(data, dump_file, ensure_ascii=False, indent=3, default=(lambda e: e.decode('utf-8')))
    dump_file.close()
