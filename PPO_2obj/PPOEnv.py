from logging import getLogger
import copy
import numpy as np
from PPODAGLoader import OffloadingDAG


class OffloadingEnvironment(object):
    def __init__(self, train_flag, **env_params):

        self.env_params = env_params
        self.logger = getLogger(name='env')
        self.train_flag = train_flag

        # system resource
        self.mec_resource = MECComputingResource(**self.env_params['mec_resource_params'])

        # data load
        if train_flag:
            list_task_graph, list_task_sequence, list_dependency_matrix = self.load_data(env_params['train_dataset'])
            self.logger.info('Train data complete')
        else:
            list_task_graph, list_task_sequence, list_dependency_matrix = self.load_data(env_params['eval_dataset'])
            self.logger.info('Eval data complete')

        self.dataset = dict(task_graph=list_task_graph,
                            task_sequence=list_task_sequence,
                            dependency_matrix=list_dependency_matrix)

        # update per batch
        self.latency = []
        self.energy = []

        # baseline
        self.logger.info('Preparing baseline...')
        self.baseline = dict()
        self.get_local_baseline()
        self.get_remote_baseline()
        self.baseline.update(local_reward=self.get_normalized_reward(self.baseline['local_le'][:, :, 0],
                                                                     self.baseline['local_le'][:, :, 1],
                                                                     self.baseline),
                             remote_reward=self.get_normalized_reward(self.baseline['remote_le'][:, :, 0],
                                                                      self.baseline['remote_le'][:, :, 1],
                                                                      self.baseline))
        self.logger.info('Baseline complete')

        if train_flag:
            self.epoch_baseline = dict.fromkeys(self.baseline)
            self.epoch_problem = None
            self.epoch_dependency = None
            self.epoch_pref = None

        # update per batch
        self.batch_idx = np.zeros(0)
        self.batch_baseline = dict.fromkeys(self.baseline)
        self.pref = None
        self.problems = None
        self.dependencies = None
        self.rewards = None

    ########################################################################
    # data loading
    ########################################################################
    def load_data(self, params):
        list_task_graph = []
        list_task_sequence = []
        list_dependency_matrix = []

        if params['load_type'] == 'gv':
            for graph_file_path in params['graph_file_path']:
                self.logger.info('loading data from gv file: {}'.format(graph_file_path))
                task_graph_list_in_file, task_sequence_list_in_file, dependency_list_in_file \
                    = self.generate_list(graph_file_path, params['graph_number_per_file'])
                list_task_graph += task_graph_list_in_file
                list_task_sequence += task_sequence_list_in_file
                list_dependency_matrix += dependency_list_in_file

            list_task_sequence = np.array(list_task_sequence)
            list_dependency_matrix = np.array(list_dependency_matrix)

            # save as preprocessed numpy file
            # np.save(params['numpy_file_path'] + '_task_sequence', list_task_sequence)
            # np.save(params['numpy_file_path'] + '_dependency_matrix', list_dependency_matrix)

        elif params['load_type'] == 'npy':
            self.logger.info('loading data from npy file path: {}'.format(params['numpy_file_path']))
            list_task_graph = None
            list_task_sequence = np.load(params['numpy_file_path'] + '_task_sequence' + '.npy')
            list_dependency_matrix = np.load(params['numpy_file_path'] + '_dependency_matrix' + '.npy')

        else:
            list_task_graph = None
            list_task_sequence = None
            list_dependency_matrix = None
            raise AssertionError('Data can only be loaded from .gv or .npy files!!!')

        return list_task_graph, list_task_sequence, list_dependency_matrix

    def init_epoch_dataset(self, **train_set):
        # check instance_num & batch_size
        instance_num = train_set['batch_size']
        if train_set['pref_flag'] == 'step':
            min_instance = int(1 / train_set['pref_step'] + 1) * instance_num
        elif train_set['pref_flag'] == 'random':
            min_instance = train_set['pref_per_epoch'] * instance_num
        else:
            raise AssertionError('Parameter \'pref_flag\' must be \'step\' or \'random\'!!!')

        dataset_num = self.dataset['task_sequence'].shape[0]
        if min_instance > dataset_num:
            problem = np.repeat(self.dataset['task_sequence'], np.ceil(min_instance/dataset_num), axis=0)
            dependency = np.repeat(self.dataset['dependency_matrix'], np.ceil(min_instance/dataset_num), axis=0)

            perm = np.arange(problem.shape[0])
            np.random.shuffle(perm)
            perm = perm[:min_instance]

            self.epoch_problem = problem[perm]
            self.epoch_dependency = dependency[perm]
            for key in self.baseline:
                baseline = np.repeat(self.baseline[key], np.ceil(min_instance/dataset_num), axis=0)
                self.epoch_baseline[key] = baseline[perm]
                self.epoch_baseline[key] = self.epoch_baseline[key][:min_instance]
        else:
            self.epoch_problem = self.dataset['task_sequence']
            self.epoch_dependency = self.dataset['dependency_matrix']
            np.random.shuffle(self.epoch_problem)
            np.random.shuffle(self.epoch_dependency)
            for key in self.baseline:
                self.epoch_baseline[key] = self.baseline[key]
                np.random.shuffle(self.epoch_baseline[key])

        self.logger.info('Train epoch instance num: {:d}'.format(self.epoch_problem.shape[0]))
        self.logger.info('PPO batch size: {:d}'.format(train_set['batch_size']))
        self.logger.info('Preference generation rule: {}'.format(train_set['pref_flag']))
        self.logger.info('Each epoch will train with {:d} sets of preferences'.format(int(
            np.ceil(self.epoch_problem.shape[0]/train_set['batch_size']))))
        self.logger.info('--------------------------------------------------------------------------')

    def init_idx(self, shuffle: bool, train_set=None):
        if self.train_flag:
            instance_num = self.epoch_problem.shape[0]
        else:
            instance_num = self.dataset['task_sequence'].shape[0]

        # generate pref list
        if train_set:
            pref_num = int(np.ceil(instance_num / train_set['batch_size']))
            if train_set['pref_flag'] == 'step':
                pref0 = np.arange(1, 0-1e-10, -train_set['pref_step'])
                pref1 = np.ones(int(1/train_set['pref_step']) + 1) - pref0
                pref = np.hstack((np.expand_dims(pref0, axis=1), np.expand_dims(pref1, axis=1)))
                self.epoch_pref = pref.repeat(np.ceil(pref_num / pref0.shape[0]), axis=0)
                self.epoch_pref = self.epoch_pref[:pref_num]
            elif train_set['pref_flag'] == 'random':
                pref0 = np.random.rand(int(pref_num))
                pref1 = np.ones(int(pref_num)) - pref0
                self.epoch_pref = np.hstack((np.expand_dims(pref0, axis=1), np.expand_dims(pref1, axis=1)))
            else:
                raise AssertionError('Parameter \'pref_flag\' in \'runner_params\' must be \'step\' or \'random\' !!!')
            np.random.shuffle(self.epoch_pref)

        # dataset shuffle
        if shuffle:
            perm = np.arange(instance_num)
            np.random.shuffle(perm)
            self.batch_idx = perm
        else:
            self.batch_idx = np.arange(instance_num)

    def load_batch(self, batch_size):
        idx = self.batch_idx[:batch_size]
        self.batch_idx = self.batch_idx[batch_size:]

        # for train
        if self.train_flag:
            # [batch, task_num, 17]
            self.problems = self.epoch_problem[idx]
            for key in self.epoch_baseline:
                self.batch_baseline[key] = self.epoch_baseline[key][idx]
            self.dependencies = self.epoch_dependency[idx]
            pref_index = int((self.epoch_problem.shape[0] - self.batch_idx.shape[0])/batch_size - 1)
            self.pref = self.epoch_pref[pref_index, :]
        # for eval
        else:
            self.problems = self.dataset['task_sequence']
            self.dependencies = self.dataset['dependency_matrix']
            for key in self.baseline:
                self.batch_baseline[key] = self.baseline[key][idx]

        self.latency = []
        self.energy = []
        self.rewards = []

    def init_batch_reward(self):
        self.latency = []
        self.energy = []
        self.rewards = []

    ########################################################################
    # reward calculation
    ########################################################################
    def get_reward(self, actions, reward_type=None):
        for action, problem, dependency in zip(actions, self.problems, self.dependencies):
            self.get_pre_reward_for_one_problem(action, problem, dependency)
        self.latency = np.array(self.latency)
        self.energy = np.array(self.energy)
        if reward_type == 'qoe':
            self.rewards = self.get_qoe_reward(self.latency, self.energy, self.batch_baseline)
        else:
            self.rewards = self.get_normalized_reward(self.latency, self.energy, self.batch_baseline)

    def get_pre_reward_for_one_problem(self, action, problem, dependency):
        time_cost = problem[:, 1:1+4].T

        matrix_channel = np.zeros(time_cost.shape)
        matrix_start_time = np.zeros(time_cost.shape)
        matrix_finish_time = np.zeros(time_cost.shape)

        local, upload, cloud, download = 0, 1, 2, 3
        latency_list = np.zeros(time_cost.shape[1])
        time_available = np.zeros(time_cost.shape[0])
        time_start = np.zeros(time_cost.shape[0])
        time_finish = np.zeros(time_cost.shape[0])
        task_finish_time = 0.0
        current_finish_time = 0.0

        energy_list = np.zeros(time_cost.shape[1])

        for i in range(problem.shape[0]):
            idx = int(problem[i, 0])
            pre_idx = dependency[:, idx].nonzero()[0]
            matrix_channel[:, idx] = time_available

            if action[i] == 0:
                if len(pre_idx) > 1:
                    time_start[local] = max(time_available[local],
                                            max([max(matrix_finish_time[local, j],
                                                     matrix_finish_time[download, j]
                                                     ) for j in pre_idx[:-1]]))
                else:
                    time_start[local] = time_available[local]

                time_finish[local] = time_start[local] + time_cost[local, i]
                time_available[local] = time_finish[local]

                task_finish_time = time_finish[local].copy()
                energy_list[i] = self.mec_resource.energy_cost(time_cost[local, i], 'local')

            elif action[i] == 1:
                if len(pre_idx) > 1:
                    time_start[upload] = max(time_available[upload],
                                             max([max(matrix_finish_time[local, j],
                                                      matrix_finish_time[download, j]
                                                      ) for j in pre_idx[:-1]]))
                else:
                    time_start[upload] = time_available[upload]

                time_finish[upload] = time_start[upload] + time_cost[upload, i]
                time_available[upload] = time_finish[upload]

                time_start[cloud] = max(time_available[cloud], time_finish[upload])
                time_finish[cloud] = time_start[cloud] + time_cost[cloud, i]
                time_available[cloud] = time_finish[cloud]

                time_start[download] = max(time_available[download], time_finish[cloud])
                time_finish[download] = time_start[download] + time_cost[download, i]
                time_available[download] = time_finish[download]

                task_finish_time = time_finish[download].copy()
                energy_list[i] = self.mec_resource.energy_cost(time_cost[upload, i], 'upload') + \
                                 self.mec_resource.energy_cost(time_cost[cloud, i], 'cloud') + \
                                 self.mec_resource.energy_cost(time_cost[download, i], 'download')
            else:
                raise AssertionError('offloading_decision must be 1 or 0')

            matrix_start_time[:, idx] = time_start
            matrix_finish_time[:, idx] = time_finish

            delta_make_span = max(task_finish_time, current_finish_time) - current_finish_time
            current_finish_time = max(task_finish_time, current_finish_time)
            latency_list[i] = delta_make_span

        self.latency.append(latency_list)
        self.energy.append(energy_list)
        return latency_list, matrix_finish_time, energy_list

    def get_normalized_reward(self, latency, energy, baseline):
        # Local_base
        avg_latency = np.expand_dims(baseline['local_le'][:, :, 0].mean(axis=-1), axis=-1)
        all_latency = np.expand_dims(baseline['local_le'][:, :, 0].sum(axis=-1), axis=-1)
        avg_energy = np.expand_dims(baseline['local_le'][:, :, 1].mean(axis=-1), axis=-1)
        all_energy = np.expand_dims(baseline['local_le'][:, :, 1].sum(axis=-1), axis=-1)

        reward_latency = (avg_latency - latency) / all_latency
        reward_energy = (avg_energy - energy) / all_energy
        rewards = np.concatenate([np.expand_dims(reward_latency, -1), np.expand_dims(reward_energy, axis=-1)], axis=-1)

        return rewards

    def get_qoe_reward(self, latency, energy, baseline):
        # Local_base
        b_latency = baseline['local_le'][:, :, 0].sum(axis=-1)
        b_energy = baseline['local_le'][:, :, 1].sum(axis=-1)

        reward_latency = (b_latency - latency.sum(axis=-1)) / b_latency
        reward_energy = (b_energy - energy.sum(axis=-1)) / b_energy
        rewards = np.concatenate([np.expand_dims(reward_latency, -1), np.expand_dims(reward_energy, axis=-1)], axis=-1)

        return rewards

    def get_local_baseline(self):
        latency_local = self.dataset['task_sequence'][:, :, 1]
        energy_local = self.mec_resource.energy_cost(latency_local, 'local')

        local_le = np.concatenate([np.expand_dims(latency_local, -1),
                                   np.expand_dims(energy_local, axis=-1)], axis=-1)
        self.baseline.update(local_le=local_le)

    def get_remote_baseline(self):
        action = np.ones(self.dataset['task_sequence'].shape[1], dtype=np.int32)
        for problem, dependency in zip(self.dataset['task_sequence'], self.dataset['dependency_matrix']):
            self.get_pre_reward_for_one_problem(action, problem, dependency)

        remote_le = np.concatenate([np.expand_dims(np.array(self.latency), -1),
                                    np.expand_dims(np.array(self.energy), axis=-1)], axis=-1)
        self.baseline.update(remote_le=remote_le)

    ########################################################################
    # for data generation
    ########################################################################
    def generate_list(self, graph_file_path, num_per_file):
        task_graph_list = []
        dependency_list = []
        time_cost_list = []
        task_sequence_list = []

        for i in range(num_per_file):
            file_name = graph_file_path + str(i) + '.gv'
            task_graph = OffloadingDAG(file_name)
            task_graph_list.append(task_graph)
            dependency_list.append(task_graph.dependency_matrix)

            time_cost = self.get_per_time_cost(task_graph)
            time_cost_list.append(time_cost)

            order = self.get_order(task_graph, time_cost)

            task_sequence = self.get_task_sequence(task_graph, time_cost, order)
            task_sequence_list.append(task_sequence)

        return task_graph_list, task_sequence_list, dependency_list

    def get_per_time_cost(self, task_graph):
        task_length = len(task_graph.offloading_decision)
        time_cost = np.array([[0.0] * task_length] * 4)
        for i in range(task_length):
            sub_task = task_graph.subtask_list[i]

            T_l = self.mec_resource.latency_cost(sub_task.processing_data_size, 'local')
            T_u = self.mec_resource.latency_cost(sub_task.processing_data_size, 'upload')
            T_e = self.mec_resource.latency_cost(sub_task.processing_data_size, 'cloud')
            T_d = self.mec_resource.latency_cost(sub_task.transmission_data_size, 'download')

            time_cost[:, i] = [T_l, T_u, T_e, T_d]

        return time_cost

    def get_order(self, task_graph, time_cost):
        task_length = len(task_graph.offloading_decision)
        greedy_cost = np.vstack((time_cost[0, :], time_cost[1:4, :].sum(0))).min(0)
        rank_dict = [-1] * task_length

        def rank(task_index):
            if rank_dict[task_index] != -1:
                return rank_dict[task_index]

            if len(task_graph.order_suc_ids[task_index]) == 0:
                rank_dict[task_index] = greedy_cost[task_index]
                return rank_dict[task_index]
            else:
                rank_dict[task_index] = greedy_cost[task_index] + \
                                        max(rank(j) for j in task_graph.order_suc_ids[task_index])
                return rank_dict[task_index]

        for i in range(task_length):
            rank(i)

        sort = np.argsort(rank_dict)[::-1]

        return sort

    def get_task_sequence(self, task_graph, time_cost, order):
        point_sequence = []
        for subtask_id in order:
            task_embedding_vector = np.hstack((subtask_id, time_cost[:, subtask_id]))

            pre_task_index_set = []
            succ_task_index_set = []

            for pre_task_index in range(0, subtask_id):
                if task_graph.dependency_matrix[pre_task_index][subtask_id] > 0.1:
                    pre_task_index_set.append(pre_task_index)

            while len(pre_task_index_set) < 6:
                pre_task_index_set.append(-1.0)

            for succ_task_index in range(subtask_id + 1, len(task_graph.subtask_list)):
                if task_graph.dependency_matrix[subtask_id][succ_task_index] > 0.1:
                    succ_task_index_set.append(succ_task_index)

            while len(succ_task_index_set) < 6:
                succ_task_index_set.append(-1.0)

            succ_task_index_set = succ_task_index_set[0:6]
            pre_task_index_set = pre_task_index_set[0:6]

            point_vector = np.hstack((task_embedding_vector, np.array(pre_task_index_set + succ_task_index_set)))
            point_sequence.append(point_vector)

        return np.array(point_sequence)


class MECComputingResource(object):
    def __init__(self, **mec_resource_params):
        self.mec_resource_params = mec_resource_params
        self.mobile_process_capable = self.mec_resource_params['mobile_process_capable']
        self.mec_process_capable = self.mec_resource_params['mec_process_capable']
        self.bandwidth_up = mec_resource_params['bandwidth_up'] * (1024.0 * 1024.0 / 8.0)
        self.bandwidth_dl = mec_resource_params['bandwidth_dl'] * (1024.0 * 1024.0 / 8.0)

        self.power_mobile = mec_resource_params['rho'] * (mec_resource_params['f_l'] ** mec_resource_params['zeta'])
        self.power_up = mec_resource_params['power_up']
        self.power_cloud = mec_resource_params['power_cloud']
        self.power_dl = mec_resource_params['power_dl']

    def latency_cost(self, data, compute_type):
        if compute_type == 'local':
            return data / self.mobile_process_capable
        if compute_type == 'cloud':
            return data / self.mec_process_capable
        if compute_type == 'upload':
            return data / self.bandwidth_up
        if compute_type == 'download':
            return data / self.bandwidth_dl
        else:
            raise AssertionError('choose compute_type!')

    def energy_cost(self, _time, compute_type):
        if compute_type == 'local':
            return _time * self.power_mobile
        if compute_type == 'upload':
            return _time * self.power_up
        if compute_type == 'cloud':
            return _time * self.power_cloud
        if compute_type == 'download':
            return _time * self.power_dl
        else:
            raise AssertionError('choose compute_type!')
