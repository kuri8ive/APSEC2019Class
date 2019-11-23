# -*- coding: utf-8 -*-

import os
import shelve
from collections import defaultdict


class Model:
    def __init__(self):
        # Placeholders
        self.model_name = ''
        self.embedding_dim = 200   # embeddingの次元数
        self.embeddings_train = None  # 全ノードのベクトルをnumpy行列としてもつ
        self.embeddings_test = None
        self.epochs = 1000
        self.decay = 'exp999'
        self.cv_index = 1
        self.accuracy = 0.0

        self.relations_matrix_train = None
        self.relations_matrix_test = None
        self.relations_matrix_test_not_be_used_from_other_class = None
        self.not_negative_sampling_candidates_index_dict_train = None
        self.not_negative_sampling_candidates_index_dict_test = None
        self.negative_sample_num = None

        self.num_of_recommend_class = None
        self.num_of_train_nodes_and_recommend_class = None
        self.num_of_test_classes = None
        self.num_of_train_classes = None
        self.num_of_classes = None
        self.num_of_methods = None
        self.num_of_fields = None
        self.num_of_all_nodes = None
        self.num_of_all_edges = None
        self.num_of_train_nodes = None
        self.num_of_test_nodes = None
        self.alpha = 0.5
        self.beta = 0.3
        self.gamma = 0.5
        self.delta = 0.3
        self.target_negative_weight = 1

        self.initial_learn_rate = 0.0

        self.number_data_1st_10th_and_failed = None
        self.recommended_names_success = None
        self.recommended_names_failed = None
        self.recommend_success_count_in_each_ranking = None
        self.success_if_related_all = 0.0
        self.success_if_related_not_all = 0.0

        self.resource = None
        self.std_out = True
        self.input_path = ''
        self.output_path = ''
        self.log_dir = ''
        self.train_ratio = None

        # 宣言クラス・メソッド・フィールド
        self.declaration_classes_train = set()
        self.declaration_classes_test = set()
        self.declaration_classes_all = set()
        self.not_declaration_classes_train = set()
        self.not_declaration_classes_test = set()
        self.not_declaration_classes_all = set()
        self.methods_in_class_train = set()
        self.methods_in_class_test = set()
        self.methods_in_class_all = set()
        self.not_methods_in_class_train = set()
        self.not_methods_in_class_test = set()
        self.not_methods_in_class_all = set()
        self.fields_in_class_train = set()
        self.fields_in_class_test = set()
        self.fields_in_class_all = set()
        self.not_fields_in_class_train = set()
        self.not_fields_in_class_test = set()
        self.not_fields_in_class_all = set()

        # 親クラス
        self.super_classes = set()
        # 呼び出されているメソッド（コンストラクタ含む），メソッド内で読み書きされているフィールド，戻り値の型
        self.callee_methods = set()
        self.fields_in_method = set()
        self.return_type_classes = set()
        # フィールドの型
        self.field_type_classes = set()

        # 何であれクラス，メソッド，フィールド
        self.classes = set()
        self.methods = set()
        self.fields = set()

        self.classes_train = set()
        self.classes_test = set()
        self.methods_train = set()
        self.methods_test = set()
        self.fields_train = set()
        self.fields_test = set()

        # クラス，メソッド，フィールド名をキー，行列のindexをvalueとする辞書
        self.name_to_index = defaultdict(int)
        self.index_to_name = defaultdict(str)

        # 全ノード集合と，確率的勾配降下法で使う用のコピー
        self.all_nodes = set()
        self.all_train_nodes = set()
        self.all_test_nodes = set()

        # クラスと継承元クラスの関係
        self.class_extends_relations = defaultdict(str)
        self.class_extends_relations_reverse = defaultdict(set)
        self.num_of_class_extends_relations = None

        # クラスとメソッドの関係
        self.method_in_class_relations = defaultdict(set)
        self.method_in_class_relations_reverse = defaultdict(str)
        self.num_of_method_in_class_relations = None

        # クラスとフィールドの関係
        self.field_in_class_relations = defaultdict(set)
        self.field_in_class_relations_reverse = defaultdict(str)
        self.num_of_field_in_class_relations = None

        # callerメソッドとcalleeメソッドの関係
        self.method_call_relations = defaultdict(set)
        self.method_call_relations_reverse = defaultdict(set)
        self.num_of_method_call_relations = None

        # メソッドと読み書きしているフィールドの関係
        self.field_in_method_relations = defaultdict(set)
        self.field_in_method_relations_reverse = defaultdict(set)
        self.num_of_field_in_method_relations = None

        # メソッドと戻り値の型クラスの関係
        self.return_type_relations = defaultdict(str)
        self.return_type_relations_reverse = defaultdict(set)
        self.num_of_return_type_relations = None

        # フィールドと型クラスの関係
        self.field_type_relations = defaultdict(str)
        self.field_type_relations_reverse = defaultdict(set)
        self.num_of_field_type_relations = None
        # self.logger.addHandler(self.handler)

        self.how_many_used_dict = None

    def save(self, output_file_path):
        """
        自身のオブジェクトをファイルに出力し永続化する
        :param output_file_path: 出力先ファイルパス
        """

        if not os.path.isdir(os.path.dirname(output_file_path)):
            os.makedirs(os.path.dirname(output_file_path))

        db = shelve.open(output_file_path)

        db['how_many_used_dict'] = self.how_many_used_dict
        db['model_name'] = self.model_name
        db['accuracy'] = self.accuracy
        db['cv_index'] = self.cv_index
        db['epochs'] = self.epochs
        db['decay'] = self.decay
        db['initial_learn_rate'] = self.initial_learn_rate
        db["resource"] = self.resource
        db["std_out"] = self.std_out
        db["input_path"] = self.input_path
        db['output_path'] = self.output_path
        db["log_dir"] = self.log_dir
        db["train_ratio"] = self.train_ratio
        db["embedding_dim"] = self.embedding_dim
        db["embeddings_train"] = self.embeddings_train
        db["embeddings_test"] = self.embeddings_test
        db["relations_matrix_train"] = self.relations_matrix_train
        db['relations_matrix_test'] = self.relations_matrix_test
        db['relations_matrix_test_not_be_used_from_other_class'] = self.relations_matrix_test_not_be_used_from_other_class
        db['not_negative_sampling_candidates_index_dict_train'] = self.not_negative_sampling_candidates_index_dict_train
        db['not_negative_sampling_candidates_index_dict_test'] = self.not_negative_sampling_candidates_index_dict_test
        db['negative_sample_num'] = self.negative_sample_num
        db['num_of_all_nodes'] = self.num_of_all_nodes
        db['num_of_all_edges'] = self.num_of_all_edges
        db['num_of_train_nodes'] = self.num_of_train_nodes
        db['num_of_test_nodes'] = self.num_of_test_nodes
        db['num_of_train_nodes_and_recommend_class'] = self.num_of_train_nodes_and_recommend_class
        db['num_of_test_classes'] = self.num_of_test_classes
        db['num_of_train_classes'] = self.num_of_train_classes
        db['alpha'] = self.alpha
        db['beta'] = self.beta
        db['gamma'] = self.gamma
        db['delta'] = self.delta
        db['declaration_classes_train'] = self.declaration_classes_train
        db['declaration_classes_test'] = self.declaration_classes_test
        db['declaration_classes_all'] = self.declaration_classes_all
        db['not_declaration_classes_train'] = self.not_declaration_classes_train
        db['not_declaration_classes_test'] = self.not_declaration_classes_test
        db['not_declaration_classes_all'] = self.not_declaration_classes_all
        db['methods_in_class_train'] = self.methods_in_class_train
        db['methods_in_class_test'] = self.methods_in_class_test
        db['methods_in_class_all'] = self.methods_in_class_all
        db['not_methods_in_class_train'] = self.not_methods_in_class_train
        db['not_methods_in_class_test'] = self.not_methods_in_class_test
        db['not_methods_in_class_all'] = self.not_methods_in_class_all
        db['fields_in_class_train'] = self.fields_in_class_train
        db['fields_in_class_test'] = self.fields_in_class_test
        db['fields_in_class_all'] = self.fields_in_class_all
        db['not_fields_in_class_train'] = self.not_fields_in_class_train
        db['not_fields_in_class_test'] = self.not_fields_in_class_test
        db['not_fields_in_class_all'] = self.not_fields_in_class_all
        db['super_classes'] = self.super_classes
        db['callee_methods'] = self.callee_methods
        db['fields_in_method'] = self.fields_in_method
        db['return_type_classes'] = self.return_type_classes
        db['field_type_classes'] = self.field_type_classes
        db['classes'] = self.classes
        db['methods'] = self.methods
        db['fields'] = self.fields
        db['classes_train'] = self.classes_train
        db['classes_test'] = self.classes_test
        db['methods_train'] = self.methods_train
        db['methods_test'] = self.methods_test
        db['fields_train'] = self.fields_train
        db['fields_test'] = self.fields_test
        db['name_to_index'] = self.name_to_index
        db['index_to_name'] = self.index_to_name
        db['all_nodes'] = self.all_nodes
        db['all_train_nodes'] = self.all_train_nodes
        db['all_test_nodes'] = self.all_test_nodes
        db['class_extends_relations'] = self.class_extends_relations
        db['class_extends_relations_reverse'] = self.class_extends_relations_reverse
        db['num_of_class_extends_relations'] = self.num_of_class_extends_relations
        db['method_in_class_relations'] = self.method_in_class_relations
        db['method_in_class_relations_reverse'] = self.method_in_class_relations_reverse
        db['num_of_method_in_class_relations'] = self.num_of_method_in_class_relations
        db['field_in_class_relations'] = self.field_in_class_relations
        db['field_in_class_relations_reverse'] = self.field_in_class_relations_reverse
        db['num_of_field_in_class_relations'] = self.num_of_field_in_class_relations
        db['method_call_relations'] = self.method_call_relations
        db['method_call_relations_reverse'] = self.method_call_relations_reverse
        db['num_of_method_call_relations'] = self.num_of_method_call_relations
        db['field_in_method_relations'] = self.field_in_method_relations
        db['field_in_method_relations_reverse'] = self.field_in_method_relations_reverse
        db['num_of_field_in_method_relations'] = self.num_of_field_in_method_relations
        db['return_type_relations'] = self.return_type_relations
        db['return_type_relations_reverse'] = self.return_type_relations_reverse
        db['num_of_return_type_relations'] = self.num_of_return_type_relations
        db['field_type_relations'] = self.field_type_relations
        db['field_type_relations_reverse'] = self.field_type_relations_reverse
        db['num_of_field_type_relations'] = self.num_of_field_type_relations
        db['number_data_1st_10th_and_failed'] = self.number_data_1st_10th_and_failed
        db['recommended_names_success'] = self.recommended_names_success
        db['recommended_names_failed'] = self.recommended_names_failed
        db['recommend_success_count_in_each_ranking'] = self.recommend_success_count_in_each_ranking
        db['num_of_recommend_class'] = self.num_of_recommend_class
        db['success_if_related_all'] = self.success_if_related_all
        db['success_if_related_not_all'] = self.success_if_related_not_all

        db.close()
        print('Trained model object was saved to {0}'.format(output_file_path))

    def reproduce(self, file_path):
        """
        save関数でファイルに出力されたモデルから各種パラメータを取得し,
        自身のインスタンスのパラメータに上書きする
        :param file_path: モデルを保存したファイルのパス
        """
        db = shelve.open(file_path)

        self.model_name = db['model_name']
        self.accuracy = db['accuracy']
        self.cv_index = db['cv_index']
        self.epochs = db['epochs']
        self.decay = db['decay']
        self.initial_learn_rate = db['initial_learn_rate']
        self.resource = db["resource"]
        self.std_out = db["std_out"]
        self.input_path = db["input_path"]
        self.output_path = db['output_path']
        self.log_dir = db["log_dir"]
        self.train_ratio = db["train_ratio"]
        self.embedding_dim = db["embedding_dim"]
        self.embeddings_train = db["embeddings_train"]
        self.embeddings_test = db["embeddings_test"]
        self.relations_matrix_train = db["relations_matrix_train"]
        self.relations_matrix_test = db['relations_matrix_test']
        self.relations_matrix_test_not_be_used_from_other_class = db['relations_matrix_test_not_be_used_from_other_class']
        self.not_negative_sampling_candidates_index_dict_train = db['not_negative_sampling_candidates_index_dict_train']
        self.not_negative_sampling_candidates_index_dict_test = db['not_negative_sampling_candidates_index_dict_test']
        self.negative_sample_num = db['negative_sample_num']
        self.num_of_all_nodes = db['num_of_all_nodes']
        self.num_of_all_edges = db['num_of_all_edges']
        self.num_of_train_nodes = db['num_of_train_nodes']
        self.num_of_test_nodes = db['num_of_test_nodes']
        self.num_of_train_nodes_and_recommend_class = db['num_of_train_nodes_and_recommend_class']
        self.num_of_test_classes = db['num_of_test_classes']
        self.num_of_train_classes = db['num_of_train_classes']
        self.alpha = db["alpha"]
        self.beta = db["beta"]
        self.gamma = db['gamma']
        self.delta = db['delta']
        self.declaration_classes_all = db['declaration_classes_all']
        self.declaration_classes_train = db['declaration_classes_train']
        self.declaration_classes_test = db['not_declaration_classes_test']
        self.not_declaration_classes_all = db['not_declaration_classes_all']
        self.not_declaration_classes_train = db['not_declaration_classes_train']
        self.not_declaration_classes_test = db['not_declaration_classes_test']
        self.methods_in_class_train = db['methods_in_class_train']
        self.methods_in_class_test = db['methods_in_class_test']
        self.methods_in_class_all = db['methods_in_class_all']
        self.not_methods_in_class_train = db['not_methods_in_class_train']
        self.not_methods_in_class_test = db['not_methods_in_class_test']
        self.not_methods_in_class_all = db['not_methods_in_class_all']
        self.fields_in_class_train = db['fields_in_class_train']
        self.fields_in_class_test = db['fields_in_class_test']
        self.fields_in_class_all = db['fields_in_class_all']
        self.not_fields_in_class_train = db['not_fields_in_class_train']
        self.not_fields_in_class_test = db['not_fields_in_class_test']
        self.not_fields_in_class_all = db['not_fields_in_class_all']
        self.super_classes = db['super_classes']
        self.callee_methods = db['callee_methods']
        self.fields_in_method = db['fields_in_method']
        self.return_type_classes = db['return_type_classes']
        self.field_type_classes = db['field_type_classes']
        self.classes = db['classes']
        self.methods = db['methods']
        self.fields = db['fields']
        self.classes_train = db['classes_train']
        self.classes_test = db['classes_test']
        self.methods_train = db['methods_train']
        self.methods_test = db['methods_test']
        self.fields_train = db['fields_train']
        self.fields_test = db['fields_test']
        self.name_to_index = db['name_to_index']
        self.index_to_name = db['index_to_name']
        self.all_nodes = db['all_nodes']
        self.all_train_nodes = db['all_train_nodes']
        self.all_test_nodes = db['all_test_nodes']
        self.class_extends_relations = db['class_extends_relations']
        self.class_extends_relations_reverse = db['class_extends_relations_reverse']
        self.num_of_class_extends_relations = db['num_of_class_extends_relations']
        self.method_in_class_relations = db['method_in_class_relations']
        self.method_in_class_relations_reverse = db['method_in_class_relations_reverse']
        self.num_of_method_in_class_relations = db['num_of_method_in_class_relations']
        self.field_in_class_relations = db['field_in_class_relations']
        self.field_in_class_relations_reverse = db['field_in_class_relations_reverse']
        self.num_of_field_in_class_relations = db['num_of_field_in_class_relations']
        self.method_call_relations = db['method_call_relations']
        self.method_call_relations_reverse = db['method_call_relations_reverse']
        self.num_of_method_call_relations = db['num_of_method_call_relations']
        self.field_in_method_relations = db['field_in_method_relations']
        self.field_in_method_relations_reverse = db['field_in_method_relations_reverse']
        self.num_of_field_in_method_relations = db['num_of_field_in_method_relations']
        self.return_type_relations = db['return_type_relations']
        self.return_type_relations_reverse = db['return_type_relations_reverse']
        self.num_of_return_type_relations = db['num_of_return_type_relations']
        self.field_type_relations = db['field_type_relations']
        self.field_type_relations_reverse = db['field_type_relations_reverse']
        self.num_of_field_type_relations = db['num_of_field_type_relations']
        self.number_data_1st_10th_and_failed = db['number_data_1st_10th_and_failed']
        self.recommended_names_success = db['recommended_names_success']
        self.recommended_names_failed = db['recommended_names_failed']
        self.how_many_used_dict = db['how_many_used_dict']
        self.recommend_success_count_in_each_ranking = db['recommend_success_count_in_each_ranking']
        self.num_of_recommend_class = db['num_of_recommend_class']
        self.success_if_related_all = db['success_if_related_all']
        self.success_if_related_not_all = db['success_if_related_not_all']

        db.close()
        print("Model was restored from {0}".format(os.path.abspath(file_path)))

    @classmethod
    def restore(cls, file_path):
        """
        save関数でファイルに出力されたモデルから各種パラメータを取得し, 同一のパラメータを持つモデルの
        インスタンスを新たに生成し, そのインスタンスを返す
        :param file_path: モデルを保存したファイルのパス
        :return: パラメータを復元したモデルのインスタンス
        """
        new_model = cls()
        new_model.reproduce(file_path)
        return new_model
