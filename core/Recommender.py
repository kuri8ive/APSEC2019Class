import os
import re
import time
from collections import defaultdict
from itertools import chain

import chainer.cuda as cuda
import cupy as cp
import numpy as np
from stemming.porter2 import stem
from tqdm import tqdm

from . import Model

THIS_FILE_DIR_PATH = os.path.dirname(os.path.abspath(__file__))


def calc_cos_similarity(a, b):
    return a.dot(b.transpose()) \
           / cp.linalg.norm(a, axis=1).reshape((a.shape[0], 1)) \
           / cp.linalg.norm(b, axis=1).reshape((1, b.shape[0]))


def calc_euclid_distance(a, b):
    all_diffs = cp.expand_dims(a, axis=1) - cp.expand_dims(b, axis=0)
    degree_distance = - cp.sqrt(np.sum(all_diffs ** 2, axis=-1))
    return degree_distance


class Recommender:
    def __init__(self, model: Model):
        self.model = model
        self.num_of_recommend_class = 0
        self.top_k_similarities = None
        self.recommend_train_similarity = None

    def recommend(self, top_k=10):
        print('recommendation started')
        start = time.time()

        train_classes = self.model.embeddings_train[:self.model.num_of_train_classes]
        recommend_classes = self.model.embeddings_test[self.model.num_of_train_nodes:self.model.num_of_train_nodes_and_recommend_class]
        self.model.num_of_recommend_class = recommend_classes.shape[0]
        print('train_classes_shape -> {}'.format(train_classes.shape))
        print('reco_class_shape -> {}'.format(recommend_classes.shape))

        self.recommend_train_similarity = calc_cos_similarity(recommend_classes, train_classes)
        self.recommend_train_similarity = cuda.to_cpu(self.recommend_train_similarity)
        self.top_k_similarities = (np.flip(np.argsort(self.recommend_train_similarity, axis=1), axis=1)[:, :top_k])

        success = 0
        success_if_related_all = success_if_related_not_all = 0
        total_if_related_all = total_if_related_not_all = 0
        self.model.number_data_1st_10th_and_failed = [defaultdict(list) for _ in range(top_k + 1)]
        self.model.recommended_names_success = defaultdict(list)
        self.model.recommended_names_failed = defaultdict(list)
        self.model.recommend_success_count_in_each_ranking = [0 for _ in range(top_k)]

        for recommend_class_index, similar_class_index_array in tqdm(enumerate(self.top_k_similarities)):
            recommend_class_name = self.model.index_to_name[self.model.num_of_train_nodes + recommend_class_index]

            have_inside = len(self.model.class_extends_relations[recommend_class_name]) \
                          + len(self.model.method_in_class_relations[recommend_class_name]) \
                          + len(self.model.field_in_class_relations[recommend_class_name])
            used_outside = len(self.model.class_extends_relations_reverse[recommend_class_name]) \
                           + len(self.model.return_type_relations_reverse[recommend_class_name]) \
                           + len(self.model.field_type_relations_reverse[recommend_class_name])
            extended = 1 if len(self.model.class_extends_relations[recommend_class_name]) else 0
            have_methods = len(self.model.method_in_class_relations[recommend_class_name])
            have_fields = len(self.model.field_in_class_relations[recommend_class_name])

            recommend_class_name_word_set = split_class_name_into_words(recommend_class_name.split('.')[-1])
            stemmed_recommend_class_name_word_set = {stem(x) for x in recommend_class_name_word_set}

            is_related_all = extended and have_methods and have_fields
            if is_related_all:
                total_if_related_all += 1
            else:
                total_if_related_not_all += 1

            for ranking_index, similar_class_index in enumerate(similar_class_index_array):
                try:
                    similar_class_index = int(similar_class_index)
                    candidate_class_name_word_set = split_class_name_into_words(self.model.index_to_name[similar_class_index].split('.')[-1])
                    stemmed_candidate_class_name_word_set = {stem(x) for x in candidate_class_name_word_set}

                    # success case
                    if len(stemmed_recommend_class_name_word_set & stemmed_candidate_class_name_word_set) != 0:
                        success += 1

                        if is_related_all:
                            success_if_related_all += 1
                        else:
                            success_if_related_not_all += 1

                        self.model.recommend_success_count_in_each_ranking[ranking_index] += 1

                        self.model.number_data_1st_10th_and_failed[ranking_index]['have_inside'].append(have_inside)
                        self.model.number_data_1st_10th_and_failed[ranking_index]['used_outside'].append(used_outside)
                        self.model.number_data_1st_10th_and_failed[ranking_index]['extended'].append(extended)
                        self.model.number_data_1st_10th_and_failed[ranking_index]['have_methods'].append(have_methods)
                        self.model.number_data_1st_10th_and_failed[ranking_index]['have_fields'].append(have_fields)

                        for i in similar_class_index_array:
                            self.model.recommended_names_success[recommend_class_name].append(self.model.index_to_name[i])
                        break

                except TypeError:
                    pass

                # failed case
                if ranking_index + 1 == top_k:
                    self.model.number_data_1st_10th_and_failed[top_k]['have_inside'].append(have_inside)
                    self.model.number_data_1st_10th_and_failed[top_k]['used_outside'].append(used_outside)
                    self.model.number_data_1st_10th_and_failed[top_k]['extended'].append(extended)
                    self.model.number_data_1st_10th_and_failed[top_k]['have_methods'].append(have_methods)
                    self.model.number_data_1st_10th_and_failed[top_k]['have_fields'].append(have_fields)

                    for i in similar_class_index_array:
                        self.model.recommended_names_failed[recommend_class_name].append(self.model.index_to_name[i])

        self.model.success_if_related_all = success_if_related_all / total_if_related_all
        self.model.success_if_related_not_all = success_if_related_not_all / total_if_related_not_all
        print(f'success_if_related_all : {self.model.success_if_related_all}')
        print(f'success_if_related_not_all : {self.model.success_if_related_not_all}')
        self.model.accuracy = success / self.top_k_similarities.shape[0]

        print('recommendation finished')
        print('elapsed_time for recommendation : {}'.format(time.time() - start))


def split_class_name_into_words(class_name: str):
    # 名前解決した場合，hoge.foo.ClassNameとなるので，ClassNameだけとりたい
    splitted_name = class_name.split('.')[-1]

    # アンダースコアで分割
    splitted_name = re.split(r"_+", splitted_name)

    # ドルマークで分割
    splitted_name = list(chain.from_iterable(map(lambda mp: re.split(r"\$+", mp), splitted_name)))

    # 2文字以上数字が並んでいた場合それを1単語として分割
    splitted_name = list(chain.from_iterable(map(_split_by_numbers, splitted_name)))

    # 大文字が3文字以上連続し，かつその後に小文字が続く場合
    # 2つ目の大文字の後を単語の区切りとする
    splitted_name = list(chain.from_iterable(map(_split_by_continuous_uppercase, splitted_name)))

    # 小文字と大文字の間を単語の区切りとする
    splitted_name = list(chain.from_iterable(map(_split_by_upper_lower, splitted_name)))

    words_lcase = set(list(map(lambda w_method: w_method.lower(), splitted_name)))
    return words_lcase


def _split_by_continuous_uppercase(string: str):
    break_indices = [0] + list(map(lambda ro: ro.span()[1] - 2, re.finditer(r"[A-Z0-9]{3,}[a-z]", string)))
    return [string[i:j] for i, j in zip(break_indices, break_indices[1:] + [None])]


def _split_by_upper_lower(string: str):
    break_indices = [0] + list(map(lambda ro: ro.span()[1] - 1, re.finditer(r"[a-z][A-Z0-9]", string)))
    return [string[i:j] for i, j in zip(break_indices, break_indices[1:] + [None])]


def _split_by_numbers(string: str):
    # break_indices = [0] + list(map(lambda ro: ro.span()[1]-1, re.finditer(r"[0-9]{2,}", string)))
    # return [string[i:j] for i, j in zip(break_indices, break_indices[1:]+[None])]
    return re.findall(r"([0-9]+|[a-zA-Z_$]+)", string)