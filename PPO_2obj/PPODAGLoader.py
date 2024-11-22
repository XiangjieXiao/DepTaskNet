import numpy as np
import pydotplus


class OffloadingSubtask(object):
    def __init__(self, subtask_id, process_data_size, transmission_data_size, depth=0):
        self.subtask_id = subtask_id
        self.processing_data_size = process_data_size
        self.transmission_data_size = transmission_data_size
        self.depth = depth
        self.subtask_pre_ids = []
        self.subtask_suc_ids = []

    def print(self):
        print("task id: {}, task type name: {} task processing data size: {}, "
              "task transmission_data_size: {}".format(self.id_name, self.type_name,
                                                       self.processing_data_size,
                                                       self.transmission_data_size))


class OffloadingDAG(object):
    def __init__(self, file_name):
        self._dot_ob = pydotplus.graphviz.graph_from_dot_file(file_name)

        self._get_subtask()

        self.dependency_matrix = []
        self.edge_list = []
        self.order_pre_ids = {}
        self.order_suc_ids = {}
        self._get_dependency()

        self._get_depth()

        self.offloading_decision = [-1] * self.subtask_list.__len__()

    def _get_subtask(self):
        subtasks = self._dot_ob.get_node_list()
        self.subtask_list = [0] * len(subtasks)

        for subtask in subtasks:
            subtask_id = int(subtask.get_name()) - 1
            processing_data_size = int(eval(subtask.obj_dict['attributes']['size']))
            transmission_data_size = int(eval(subtask.obj_dict['attributes']['expect_size']))

            self.subtask_list[subtask_id] = OffloadingSubtask(subtask_id, processing_data_size, transmission_data_size)

    def _get_dependency(self):
        edges = self._dot_ob.get_edge_list()
        subtask_num = len(self.subtask_list)
        dependency_matrix = np.zeros(shape=(subtask_num, subtask_num), dtype=np.float32)

        for i in range(len(self.subtask_list)):
            self.order_pre_ids[i] = []
            self.order_suc_ids[i] = []
            dependency_matrix[i][i] = self.subtask_list[i].processing_data_size

        for edge in edges:
            source_id = int(edge.get_source()) - 1
            destination_id = int(edge.get_destination()) - 1
            data_size = int(eval(edge.obj_dict['attributes']['size']))

            self.order_pre_ids[destination_id].append(source_id)
            self.subtask_list[destination_id].subtask_pre_ids.append(source_id)
            self.order_suc_ids[source_id].append(destination_id)
            self.subtask_list[source_id].subtask_suc_ids.append(destination_id)

            self.edge_list.append([source_id, destination_id, data_size])

            dependency_matrix[source_id][destination_id] = data_size

        self.dependency_matrix = dependency_matrix

    def _get_depth(self):
        ids_to_depth = dict()

        def calculate_depth_value(id):
            if id in ids_to_depth.keys():
                return ids_to_depth[id]
            else:
                if len(self.order_pre_ids[id]) != 0:
                    depth = 1 + max([calculate_depth_value(pre_task_id) for
                                     pre_task_id in self.order_pre_ids[id]])
                else:
                    depth = 0

                ids_to_depth[id] = depth

            return ids_to_depth[id]

        for id in range(len(self.subtask_list)):
            ids_to_depth[id] = calculate_depth_value(id)

        for id, depth in ids_to_depth.items():
            self.subtask_list[id].depth = depth


if __name__ == "__main__":
    gv_file_name = '../dataset/Origin dataset/task20/random.20.2.gv'
    graph = OffloadingDAG(gv_file_name)