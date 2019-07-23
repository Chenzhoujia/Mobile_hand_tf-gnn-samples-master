from collections import namedtuple
from typing import Any, Dict, Tuple, List, Iterable

import tensorflow as tf
import numpy as np
from dpu_utils.utils import RichPath

from .sparse_graph_task import Sparse_Graph_Task, DataFold, MinibatchData
from utils import MLP
import pickle
from tqdm import tqdm
GraphSample = namedtuple('GraphSample', ['adjacency_lists',
                                         'type_to_node_to_num_incoming_edges',
                                         'node_features',
                                         'target_values',
                                         ])


class Hand_Task(Sparse_Graph_Task):
    # These magic constants were obtained during dataset generation, as result of normalising
    # the values of target properties:
    CHEMICAL_ACC_NORMALISING_FACTORS = [0.066513725, 0.012235489, 0.071939046,
                                        0.033730778, 0.033486113, 0.004278493,
                                        0.001330901, 0.004165489, 0.004128926,
                                        0.00409976, 0.004527465, 0.012292586,
                                        0.037467458]

    @classmethod
    def default_params(cls):
        params = super().default_params()
        params.update({
            'task_ids': [0],

            'add_self_loop_edges': True,
            'tie_fwd_bkwd_edges': True,
            'use_graph': True,
            'activation_function': "tanh",
            'out_layer_dropout_keep_prob': 1.0,
        })
        return params

    @staticmethod
    def name() -> str:
        return "HAND_GEN"

    @staticmethod
    def default_data_path() -> str:
        return "data/hand_gen"

    def __init__(self, params: Dict[str, Any]):
        super().__init__(params)

        # Things that will be filled once we load data:
        self.__num_edge_types = 0
        self.__annotation_size = 0

    def get_metadata(self) -> Dict[str, Any]:
        metadata = super().get_metadata()
        metadata['num_edge_types'] = self.__num_edge_types
        metadata['annotation_size'] = self.__annotation_size
        return metadata

    def restore_from_metadata(self, metadata: Dict[str, Any]) -> None:
        super().restore_from_metadata(metadata)
        self.__num_edge_types = metadata['num_edge_types']
        self.__annotation_size = metadata['annotation_size']

    @property
    def num_edge_types(self) -> int:
        return self.__num_edge_types

    @property
    def initial_node_feature_size(self) -> int:
        return self.__annotation_size

    # -------------------- Data Loading --------------------
    def load_data(self, path: RichPath) -> None:
        self._loaded_data[DataFold.TRAIN] = self.__load_data(path.join("hand_train.pkl"))
        self._loaded_data[DataFold.VALIDATION] = self.__load_data(path.join("hand_test.pkl"))

    def load_eval_data_from_path(self, path: RichPath) -> Iterable[Any]:
        if path.path == self.default_data_path():
            path = path.join("hand_test.pkl")
        return self.__load_data(path)

    def __load_data(self, data_file: RichPath) -> List[GraphSample]:
        print(" Loading hand data from %s." % (data_file,))
        data_file = "%s" % (data_file,)
        with open(data_file, 'rb') as f:
            data = pickle.load(f)

        # Get some common data out:
        num_fwd_edge_types = 0
        for g in data:
            num_fwd_edge_types = max(num_fwd_edge_types, max([e[1] for e in g['graph']]))
        if self.params['add_self_loop_edges']:
            num_fwd_edge_types += 1
        self.__num_edge_types = max(self.num_edge_types,
                                    num_fwd_edge_types * (1 if self.params['tie_fwd_bkwd_edges'] else 2))
        self.__annotation_size = max(self.__annotation_size, len(data[0]["node_features"][0]))
        return self.__process_raw_graphs(data)

    def __process_raw_graphs(self, raw_data: Iterable[Any]) -> List[GraphSample]:
        processed_graphs = []
        for d in tqdm(raw_data):
            (type_to_adjacency_list, type_to_num_incoming_edges) = \
                self.__graph_to_adjacency_lists(d['graph'], num_nodes=len(d["node_features"]))
            processed_graphs.append(
                GraphSample(adjacency_lists=type_to_adjacency_list,
                            type_to_node_to_num_incoming_edges=type_to_num_incoming_edges,
                            node_features=d["node_features"],
                            target_values=d["targets"],
                            ))
        return processed_graphs

    def __graph_to_adjacency_lists(self, graph: Iterable[Tuple[int, int, int]], num_nodes: int) \
            -> Tuple[List[np.ndarray], np.ndarray]:
        type_to_adj_list = [[] for _ in range(self.num_edge_types)]  # type: List[List[Tuple[int, int]]]
        type_to_num_incoming_edges = np.zeros(shape=(self.num_edge_types, num_nodes,))
        for src, e, dest in graph:
            if self.params['add_self_loop_edges']:
                fwd_edge_type = e  # 0 will be the self-loop type
            else:
                fwd_edge_type = e - 1  # Make edges start from 0
            type_to_adj_list[fwd_edge_type].append((src, dest))
            type_to_num_incoming_edges[fwd_edge_type, dest] += 1
            if self.params['tie_fwd_bkwd_edges']:
                type_to_adj_list[fwd_edge_type].append((dest, src))
                type_to_num_incoming_edges[fwd_edge_type, src] += 1

        if self.params['add_self_loop_edges']:
            # Add self-loop edges (idx 0, which isn't used in the data):
            for node in range(num_nodes):
                type_to_num_incoming_edges[0, node] = 1
                type_to_adj_list[0].append((node, node))

        type_to_adj_list = [np.array(sorted(adj_list), dtype=np.int32) if len(adj_list) > 0 else np.zeros(shape=(0, 2), dtype=np.int32)
                            for adj_list in type_to_adj_list]

        # Add backward edges as an additional edge type that goes backwards:
        if not (self.params['tie_fwd_bkwd_edges']):
            type_to_adj_list = type_to_adj_list[:self.num_edge_types // 2]  # We allocated too much earlier...
            for (edge_type, adj_list) in enumerate(type_to_adj_list):
                bwd_edge_type = self.num_edge_types // 2 + edge_type
                type_to_adj_list.append(np.array(sorted((y, x) for (x, y) in adj_list), dtype=np.int32))
                for (x, y) in adj_list:
                    type_to_num_incoming_edges[bwd_edge_type][y] += 1

        return type_to_adj_list, type_to_num_incoming_edges

    def make_task_input_model(self,
                              placeholders: Dict[str, tf.Tensor],
                              model_ops: Dict[str, tf.Tensor],
                              ) -> None:
        """
        通过reshape保证对batchsize的扩展性
        """
        placeholders['initial_node_features'] = \
            tf.placeholder(dtype=tf.float32, shape=[None, self.initial_node_feature_size], name='initial_node_features')
        #先用常量对初始特征进行mask。
        placeholders['adjacency_lists'] = \
            [tf.placeholder(dtype=tf.int32, shape=[None, 2], name='adjacency_e%s' % e)
                for e in range(self.num_edge_types)]
        placeholders['type_to_num_incoming_edges'] = \
            tf.placeholder(dtype=tf.float32, shape=[self.num_edge_types, None], name='type_to_num_incoming_edges')

        model_ops['initial_node_features'] = placeholders['initial_node_features']
        model_ops['adjacency_lists'] = placeholders['adjacency_lists']
        model_ops['type_to_num_incoming_edges'] = placeholders['type_to_num_incoming_edges']

        # 在图结构中做数据处理是不合理的
        select = tf.constant([0,6,12,18,24,30,31],dtype=tf.float32)#与 json_gen中保持一致[0,6,12,18,24,30,31]
        model_ops['initial_node_features_select'] = select
        # point_num = 32
        # select_point_num = 7
        # with tf.variable_scope("select"):
        #     # select = tf.reshape( placeholders['initial_node_features'], [-1, point_num, 3])
        #     # select = tf.reshape( select, [-1, point_num*3])
        #     # with tf.variable_scope("regression_gate"):
        #     #     regression_gate = \
        #     #         MLP(point_num * 3, 100, [], 1)
        #     # with tf.variable_scope("regression"):
        #     #     regression_transform = \
        #     #         MLP(100, select_point_num, [], 1)
        #     #
        #     # select = regression_gate(select)
        #     # select = regression_transform(select)
        #     # select = tf.minimum(tf.maximum(select, -1), 1)
        #     # select = (select + 1) / 2 * (point_num - 1)
        #     # select = tf.reduce_mean(select, axis=0)  # [b,5]
        #     # select = tf.round(select)

        # mask = tf.expand_dims(tf.range(point_num), 1)
        # mask = tf.cast(tf.tile(mask, [1, 3]), tf.float32)
        # ones_mask = tf.ones_like(mask)
        # zeros_mask = tf.zeros_like(mask)
        #
        # for point in range(select_point_num):
        #     if point == 0:
        #         mask_log = tf.equal(mask - select[point], zeros_mask)
        #     else:
        #         mask_log = tf.logical_or(mask_log, tf.equal(mask - select[point], zeros_mask))
        #
        # select = tf.where(mask_log, ones_mask, zeros_mask)
        # source_data = tf.reshape(model_ops['initial_node_features'], [-1, point_num, 3])
        # source_data = tf.multiply(source_data, select)
        # model_ops['initial_node_features'] = tf.reshape(source_data, [-1, 3])

    # -------------------- Model Construction --------------------
    def make_task_output_model(self,
                               placeholders: Dict[str, tf.Tensor],
                               model_ops: Dict[str, tf.Tensor],
                               ) -> None:
        placeholders['graph_nodes_list'] = \
            tf.placeholder(dtype=tf.int32, shape=[None], name='graph_nodes_list')
        placeholders['target_values'] = \
            tf.placeholder(dtype=tf.float32, shape=[None, 3], name='target_values')
        # placeholders['out_layer_dropout_keep_prob'] = \
        #     tf.placeholder(dtype=tf.float32, shape=[], name='out_layer_dropout_keep_prob')

        task_metrics = {}
        losses = []
        loss_pre = []
        final_node_feature_size = model_ops['final_node_representations'].shape.as_list()[-1]
        for (internal_id, task_id) in enumerate(self.params['task_ids']):
            with tf.variable_scope("out_layer_task%i" % task_id):
                with tf.variable_scope("regression_gate"):
                    regression_gate = \
                        MLP(final_node_feature_size, 3, [], 1.0)
                            # placeholders['out_layer_dropout_keep_prob'])
                with tf.variable_scope("regression"):
                    regression_transform = \
                        MLP(final_node_feature_size, 3, [], 1.0)
                            # placeholders['out_layer_dropout_keep_prob'])

                per_node_outputs = regression_transform(model_ops['final_node_representations'])
                gate_input = model_ops['final_node_representations']
                per_node_gated_outputs = tf.nn.sigmoid(regression_gate(gate_input)) * per_node_outputs

                # 计算loss
                per_graph_outputs = per_node_gated_outputs
                model_ops['final_output_node_representations'] = per_graph_outputs
                per_graph_outputs = tf.add(per_graph_outputs, 0, name='final_output_node_representations')

                per_graph_errors = per_graph_outputs - placeholders['target_values']
                task_metrics['abs_err_task%i' % task_id] = tf.reduce_sum(tf.abs(per_graph_errors))
                # 计算逐关节loss
                per_graph_errors_per = tf.square(per_graph_errors)
                per_graph_errors_per = tf.reduce_mean(per_graph_errors_per,axis=-1)
                per_graph_errors_per = tf.reshape(per_graph_errors_per,[-1,32])
                per_graph_errors_per = tf.reduce_mean(per_graph_errors_per, axis=0)
                loss_pre.append(per_graph_errors_per)
                tf.summary.scalar('mae_task%i' % task_id,
                                  task_metrics['abs_err_task%i' % task_id] / tf.cast(placeholders['num_graphs'], tf.float32))
                losses.append(tf.reduce_mean(0.5 * tf.square(per_graph_errors)))
        model_ops['task_metrics'] = task_metrics
        model_ops['task_metrics']['loss'] = tf.reduce_sum(losses)
        model_ops['task_metrics']['pre_loss'] = loss_pre[-1]
        model_ops['task_metrics']['total_loss'] = model_ops['task_metrics']['loss'] * tf.cast(placeholders['num_graphs'], tf.float32)

    # -------------------- Minibatching and training loop --------------------
    def make_minibatch_iterator(self,
                                data: Iterable[Any],
                                data_fold: DataFold,
                                model_placeholders: Dict[str, tf.Tensor],
                                max_nodes_per_batch: int) \
            -> Iterable[MinibatchData]:
        if data_fold == DataFold.TRAIN:
            np.random.shuffle(data)
            out_layer_dropout_keep_prob = self.params['out_layer_dropout_keep_prob']
        else:
            out_layer_dropout_keep_prob = 1.0

        # Pack until we cannot fit more graphs in the batch
        num_graphs = 0
        while num_graphs < len(data):
            num_graphs_in_batch = 0
            batch_node_features = []  # type: List[np.ndarray]
            batch_target_task_values = []
            batch_adjacency_lists = [[] for _ in range(self.num_edge_types)]  # type: List[List[np.ndarray]]
            batch_type_to_num_incoming_edges = []
            batch_graph_nodes_list = []
            node_offset = 0

            while num_graphs < len(data) and node_offset + len(data[num_graphs].node_features) < max_nodes_per_batch:
                cur_graph = data[num_graphs]
                num_nodes_in_graph = len(cur_graph.node_features)
                batch_node_features.extend(cur_graph.node_features)
                batch_target_task_values.extend(cur_graph.target_values)
                batch_graph_nodes_list.append(np.full(shape=[num_nodes_in_graph],
                                                      fill_value=num_graphs_in_batch,
                                                      dtype=np.int32))
                for i in range(self.num_edge_types):
                    batch_adjacency_lists[i].append(cur_graph.adjacency_lists[i] + node_offset)

                # Turn counters for incoming edges into np array:
                batch_type_to_num_incoming_edges.append(cur_graph.type_to_node_to_num_incoming_edges)

                num_graphs += 1
                num_graphs_in_batch += 1
                node_offset += num_nodes_in_graph

            batch_feed_dict = {
                model_placeholders['initial_node_features']: np.array(batch_node_features),
                model_placeholders['type_to_num_incoming_edges']: np.concatenate(batch_type_to_num_incoming_edges, axis=1),
                model_placeholders['graph_nodes_list']: np.concatenate(batch_graph_nodes_list),
                model_placeholders['target_values']: np.array(batch_target_task_values),
                # model_placeholders['out_layer_dropout_keep_prob']: out_layer_dropout_keep_prob,
            }

            # Merge adjacency lists:
            num_edges = 0
            for i in range(self.num_edge_types):
                if len(batch_adjacency_lists[i]) > 0:
                    adj_list = np.concatenate(batch_adjacency_lists[i])
                else:
                    adj_list = np.zeros((0, 2), dtype=np.int32)
                num_edges += adj_list.shape[0]
                batch_feed_dict[model_placeholders['adjacency_lists'][i]] = adj_list

            yield MinibatchData(feed_dict=batch_feed_dict,
                                num_graphs=num_graphs_in_batch,
                                num_nodes=node_offset,
                                num_edges=num_edges)

    def early_stopping_metric(self, task_metric_results: List[Dict[str, np.ndarray]], num_graphs: int) -> float:
        # Early stopping based on average loss:
        return np.sum([m['total_loss'] for m in task_metric_results]) / num_graphs

    def pretty_print_epoch_task_metrics(self, task_metric_results: List[Dict[str, np.ndarray]], num_graphs: int) -> str:
        maes = {}
        for task_id in self.params['task_ids']:
            maes['mae_task%i' % task_id] = 0.
        fnum_graphs = float(num_graphs)
        for batch_task_metric_results in task_metric_results:
            for task_id in self.params['task_ids']:
                maes['mae_task%i' % task_id] += batch_task_metric_results['abs_err_task%i' % task_id] / fnum_graphs

        maes_str = " ".join("%i:%.5f" % (task_id, maes['mae_task%i' % task_id])
                            for task_id in self.params['task_ids'])
        # The following translates back from MAE on the property values normalised to the [0,1] range to the original scale:
        err_str = " ".join("%i:%.5f" % (task_id, maes['mae_task%i' % task_id] / self.CHEMICAL_ACC_NORMALISING_FACTORS[task_id])
                           for task_id in self.params['task_ids'])

        return "MAEs: %s | Error Ratios: %s" % (maes_str, err_str)
