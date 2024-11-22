import copy
import time
import os, sys
from logging import getLogger

import numpy as np

import torch
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
import torch.nn.functional as F

from utils import *
from PPOMixEnv import OffloadingEnvironment as Env
from PPOMixModel import ActorCritic


class PPOMemory:
    def __init__(self, task_class_num):
        self.class_num = task_class_num
        self.data_map = None
        self.loss_map = None
        self._instance_idx = None
        self._class_idx = None
        self._num_instance = None
        self._episode = None

    def push(self, task_class_idx, data_map):
        if self.data_map is None:
            self.data_map = dict.fromkeys(data_map)
            for key in data_map:
                self.data_map[key] = [[] for x in range(self.class_num)]
                self.data_map[key][task_class_idx].append(data_map[key])
        else:
            for key in data_map:
                self.data_map[key][task_class_idx].append(data_map[key])

    def get_mean_reward(self):
        mean_rewards = 0
        for key in self.data_map:
            for i in range(self.class_num):
                self.data_map[key][i] = np.vstack(self.data_map[key][i])
                if key == 'rewards':
                    mean_rewards += self.data_map[key][i].sum(-1).mean()
        return mean_rewards/self.class_num

    def push_loss(self, loss_map):
        if self.loss_map is None:
            for key in loss_map:
                loss_map[key] = loss_map[key].detach().cpu().numpy()
            self.loss_map = loss_map
        else:
            for key in loss_map:
                self.loss_map[key] = np.append(self.loss_map[key], loss_map[key].detach().cpu().numpy())

    def iterate_init(self):
        num_instance = 0
        instance_idx = []
        for i in range(self.class_num):
            instance_idx.append([i for i in range(len(self.data_map['problems'][i]))])
            num_instance += len(self.data_map['problems'][i])

        self._instance_idx = instance_idx
        self._class_idx = [i for i in range(self.class_num)]
        self._num_instance = num_instance
        self._episode = 0

    def get_batch(self, batch_size):
        class_idx = np.random.choice(self._class_idx)
        instance_idx = self._instance_idx[class_idx][:batch_size]

        batch = dict.fromkeys(self.data_map)
        for key in self.data_map:
            batch[key] = self.data_map[key][class_idx][instance_idx]

        self._instance_idx[class_idx] = self._instance_idx[class_idx][batch_size:]

        if len(self._instance_idx[class_idx]) == 0:
            self._class_idx.remove(class_idx)

        self._episode += len(batch['problems'])

        return batch

    def iterate_once(self, batch_size):
        while self._episode < self._num_instance:
            yield self.get_batch(batch_size)

    def clear(self):
        self.data_map = None
        self.loss_map = None


class MECRunner:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 runner_params,
                 logger_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.runner_params = runner_params
        self.logger_params = logger_params

        # cuda
        if self.runner_params['use_cuda']:
            cuda_device_num = self.runner_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        self.skip = self.runner_params['skip_train']

        # log
        self.data_log = {}
        self.save_path = None
        self.logger = getLogger(name='runner')

        # Main Components
        self.logger.info('loading data to env......')
        self.logger.info('--------------------------------------------------------------------------')
        self.env = Env(train_flag=True, **self.env_params)
        self.env.get_train_setting(**self.runner_params['pref_set'])
        self.logger.info('--------------------------------------------------------------------------')
        self.env_eval = Env(train_flag=False, **self.env_params)
        self.logger.info('--------------------------------------------------------------------------')
        self.logger.info('Environment complete')

        self.memory = PPOMemory(task_class_num=len(self.env.dataset['task_sequence']))

        self.actor_old = ActorCritic(**self.model_params)
        self.actor_new = ActorCritic(**self.model_params)
        self.optimizer = Optimizer(self.actor_new.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        if runner_params['model_path'] is not None:
            self.logger.info('Loading model from: {}'.format(runner_params['model_path']))
            self.actor_old.load_state_dict(torch.load(runner_params['model_path']))
            self.actor_new.load_state_dict(torch.load(runner_params['model_path']))

    def set_save_path(self, folder):
        self.save_path = folder

    def log_data(self, epoch_data):
        for key in epoch_data:
            if key in self.data_log:
                if 'eval' in key:
                    for class_idx in range(len(self.data_log[key])):
                        if self.data_log[key][class_idx].shape == epoch_data[key][class_idx].shape:
                            self.data_log[key][class_idx] = \
                                np.stack([self.data_log[key][class_idx], epoch_data[key][class_idx]], axis=0)
                        else:
                            self.data_log[key][class_idx] = \
                                np.vstack([self.data_log[key][class_idx],
                                           np.expand_dims(epoch_data[key][class_idx], axis=0)])

                else:
                    self.data_log[key] = np.vstack((self.data_log[key], epoch_data[key]))
            else:
                self.data_log[key] = epoch_data[key]

    #########################################################################################
    # PPO
    #########################################################################################
    def run_ppo(self):
        self.logger.info('--------------------------------------------------------------------------')
        for epoch in range(1, self.runner_params['epochs'] + 1):
            if self.skip is False:
                self.logger.info('Start Epoch {:3d}'.format(epoch))

                # sample
                start_time = time.time()
                self.memory.clear()
                self.env.init_epoch(train_set=self.runner_params['pref_set'])
                self.sample()
                mean_reward = self.memory.get_mean_reward()
                self.logger.info("sample time cost: {:.4f} s".format(time.time() - start_time))
                self.logger.info('Sample Reward: {:.4f}'.format(mean_reward))

                # update
                self.logger.info(fmt_row(13, ['total_loss', 'policy_loss', 'value_loss', 'policy_entropy']))
                start_time = time.time()
                for _ in range(self.runner_params['sample_reuse']):
                    self.memory.iterate_init()
                    for batch in self.memory.iterate_once(self.runner_params['pref_set']['batch_size']):
                        self.update(problem=batch['problems'],
                                    sample_action=batch['actions'],
                                    sample_value=batch['values'],
                                    sample_advantage=batch['advantages'],
                                    sample_tdlamret=batch['tdlamret'],
                                    pref=batch['pref'])

                    self.logger.info(fmt_row(13, [self.memory.loss_map['total_loss'].mean(),
                                                  self.memory.loss_map['pg_loss'].mean(),
                                                  self.memory.loss_map['vf_loss'].mean(),
                                                  self.memory.loss_map['entropy'].mean()]))

                self.actor_old.load_state_dict(copy.deepcopy(self.actor_new.state_dict()))
                self.scheduler.step()
                self.log_data(dict(epoch=np.array(epoch),
                                   rewards=mean_reward,
                                   update_policy_loss=self.memory.loss_map['pg_loss'].mean(),
                                   update_value_loss=self.memory.loss_map['vf_loss'].mean(),
                                   update_entropy_loss=self.memory.loss_map['entropy'].mean()))
                self.logger.info("update time cost: {:.4f} s".format((time.time() - start_time)))

            # for eval & plot
            if (epoch == 1) | (epoch % 20 == 0):
                self.eval()

            torch.save(self.actor_old.state_dict(), self.save_path + 'model.pth')
            np.save(self.save_path + 'data_log.npy', self.data_log)
            self.logger.info('---------------------------------------------------------------------')

    def sample(self):
        self.actor_old.set_decode_type(self.actor_old.model_params['seq2seq_params']['decode_type'])
        while len(self.env.epoch_idx[-1]) > 0:
            task_choose = self.env.load_batch(self.runner_params['pref_set']['batch_size'])
            data = torch.tensor(self.env.problems, dtype=torch.float32)
            pref = torch.tensor(self.env.pref, dtype=torch.float32)

            probs, actions, logits, value = self.actor_old(data, pref)

            self.env.get_reward(copy.deepcopy(actions))
            rewards = (self.env.pref * self.env.rewards).sum(-1)

            advantages, tdlamret, _ = self.get_GAE(torch.tensor(rewards, dtype=torch.float32),
                                                   value)

            self.memory.push(task_class_idx=task_choose,
                             data_map=dict(problems=self.env.problems,
                                           actions=actions.detach().cpu().numpy(),
                                           values=value.detach().cpu().numpy(),
                                           rewards=rewards,
                                           advantages=advantages.detach().cpu().numpy(),
                                           tdlamret=tdlamret.detach().cpu().numpy(),
                                           pref=np.expand_dims(self.env.pref, axis=0).repeat(rewards.shape[0], axis=0)))

    def update(self, problem, sample_action, sample_value, sample_tdlamret, sample_advantage, pref):
        if ~(np.all(pref == pref[0])):
            raise AssertionError('batch pref must be same!')

        self.actor_old.set_decode_type('teach_forcing')
        self.actor_new.set_decode_type('teach_forcing')

        data = torch.tensor(problem, dtype=torch.float32)
        decoder_input = torch.tensor(sample_action, dtype=torch.int64)
        self.actor_old.decoder_input = self.actor_new.decoder_input = decoder_input
        _, _, logits_old, _ = self.actor_old(data, torch.tensor(pref[0], dtype=torch.float32))
        _, _, logits_new, value = self.actor_new(data, torch.tensor(pref[0], dtype=torch.float32))

        pg_loss = self.get_pg_loss(logits_old=logits_old.detach(),
                                   logits_new=logits_new,
                                   action=torch.tensor(sample_action, dtype=torch.int64),
                                   advantage=torch.tensor(sample_advantage))

        vf_loss = self.get_vf_loss(vpred=value,
                                   oldvpred=torch.tensor(sample_value),
                                   returns=torch.tensor(sample_tdlamret))

        entropy = self.get_entropy(logits_new)

        loss = pg_loss - vf_loss + entropy

        self.actor_new.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_new.parameters(), self.runner_params['max_grad_clip'])
        self.optimizer.step()

        self.memory.push_loss(dict(total_loss=loss,
                                   pg_loss=-pg_loss,
                                   vf_loss=-vf_loss,
                                   entropy=-entropy))

    def eval(self):
        start_time = time.time()

        eval_pref = []
        eval_result = []
        eval_reward = []

        pref_list = np.arange(0, 1 + 1e-10, self.runner_params['eval_pref_step'])
        pref_list = np.vstack([np.ones(pref_list.shape) - pref_list, pref_list]).T

        for class_idx in range(len(self.env_eval.dataset['task_sequence'])):

            l_e = []
            reward = []
            for pref in pref_list:

                self.env_eval.init_batch_reward()

                self.env_eval.problems = copy.deepcopy(self.env_eval.dataset['task_sequence'][class_idx])
                self.env_eval.dependencies = copy.deepcopy(self.env_eval.dataset['dependency_matrix'][class_idx])
                for key in self.env_eval.baseline:
                    self.env_eval.batch_baseline[key] = copy.deepcopy(self.env_eval.baseline[key][class_idx])

                data = torch.tensor(self.env_eval.problems, dtype=torch.float32)
                self.actor_old.set_decode_type('greedy')
                # self.actor_old.set_decode_type('multinomial')
                _, actions, _, _ = self.actor_old(data, torch.tensor(pref, dtype=torch.float32))
                self.env_eval.get_reward(copy.deepcopy(actions), reward_type='qoe')

                l_e.append(np.stack([self.env_eval.latency, self.env_eval.energy], axis=-1))
                reward.append((pref * self.env_eval.rewards).sum(-1).mean())

            eval_pref.append(pref_list)
            eval_result.append(np.array(l_e))
            eval_reward.append(np.array(reward))

        self.logger.info("Test in Eval dateset: {:.4f} s".format((time.time() - start_time)))

        eval_log = dict(eval_pref=eval_pref,
                        eval_greedy_decode_result=eval_result,
                        eval_greedy_decode_qoe=eval_reward,
                        eval_local_result=copy.deepcopy(self.env_eval.baseline['local_le']),
                        eval_remote_result=copy.deepcopy(self.env_eval.baseline['remote_le']))

        self.printer(**eval_log)

        self.log_data(eval_log)

    def printer(self, **eval_log):
        mean_reward = 0
        for class_idx in range(len(eval_log['eval_pref'])):
            greedy_decode = eval_log['eval_greedy_decode_result'][class_idx].sum(2).mean(1)
            greedy_decode_qoe = eval_log['eval_greedy_decode_qoe'][class_idx]
            mid_idx = len(greedy_decode_qoe) // 2
            local = eval_log['eval_local_result'][class_idx].sum(1).mean(0)
            remote = eval_log['eval_remote_result'][class_idx].sum(1).mean(0)

            self.logger.info('-------------------------------------------')
            self.logger.info('Task {:d} eval result:'.format(eval_log['eval_greedy_decode_result'][class_idx].shape[2]))
            self.logger.info(fmt_row(12, ['Class', 'Latency', 'Energy']))
            self.logger.info(fmt_row(12, ['1:0', greedy_decode[0, 0], greedy_decode[0, 1]]))
            self.logger.info(fmt_row(12, ['Local', local[0], local[1]]))
            self.logger.info(fmt_row(12, ['0.5:0.5', greedy_decode[mid_idx, 0], greedy_decode[mid_idx, 1]]))
            self.logger.info(fmt_row(12, ['0:1', greedy_decode[-1, 0], greedy_decode[-1, 1]]))
            self.logger.info(fmt_row(12, ['Remote', remote[0], remote[1]]))
            self.logger.info('Task {:d} eval reward is: {:.4f}'.format(
                eval_log['eval_greedy_decode_result'][class_idx].shape[2], greedy_decode_qoe.mean()))

            mean_reward += greedy_decode_qoe.mean()
        self.logger.info('-------------------------------------------')

        self.logger.info('Eval Reward: {:.4f}'.format(mean_reward / len(eval_log['eval_pref'])))
        self.logger.info('-------------------------------------------')

    #########################################################################################
    # PPO loss function
    #########################################################################################
    def get_GAE(self, reward, value):
        value_temp = torch.cat([value, torch.zeros([value.size(0), 1])], dim=1)
        td_error = reward + self.runner_params['gamma'] * value_temp[:, 1:] - value

        adv = torch.zeros([value.size(0), 1])
        last_gae_lam = torch.zeros([value.size(0), 1])

        for step in reversed(range(value.size(1))):
            last_gae_lam = td_error[:, step:step + 1] + \
                           self.runner_params['gamma'] * self.runner_params['lambda'] * last_gae_lam
            adv = torch.cat([last_gae_lam, adv], dim=1)

        advantage = adv[:, :-1]
        tdlamret = advantage + value

        return advantage, tdlamret, td_error

    def get_pg_loss(self, logits_old, logits_new, action, advantage):
        # policy gradient loss
        neg_logp_old = F.cross_entropy(input=logits_old.flatten(start_dim=0, end_dim=1),
                                       target=action.flatten(start_dim=0, end_dim=1), reduction='none')
        logp_old = -neg_logp_old.reshape(action.shape)
        neg_logp_new = F.cross_entropy(input=logits_new.flatten(start_dim=0, end_dim=1),
                                       target=action.flatten(start_dim=0, end_dim=1), reduction='none')
        logp_new = -neg_logp_new.reshape(action.shape)

        ratio = torch.exp(logp_new - logp_old)

        clip_range = self.runner_params['clip_range']

        advantage_std = (advantage - advantage.mean(dim=0)) / (advantage.std(dim=0) + 1e-8)
        pg_loss1 = -advantage_std * ratio
        pg_loss2 = -advantage_std * torch.clip(ratio, 1.0 - clip_range, 1.0 + clip_range)

        pg_loss = torch.maximum(pg_loss1, pg_loss2).mean()
        return pg_loss

    def get_vf_loss(self, vpred, oldvpred, returns):
        # value function loss
        clip_range = self.runner_params['clip_range']
        vpred_clipped = oldvpred + torch.clip(vpred - oldvpred, -clip_range, clip_range)

        vl_loss1 = torch.square(vpred - returns)
        vl_loss2 = torch.square(vpred_clipped - returns)

        vf_loss = - torch.maximum(vl_loss1, vl_loss2).mean()

        return self.runner_params['vf_coef'] * vf_loss

    def get_entropy(self, decoder_logits):
        prob = torch.softmax(decoder_logits, -1)
        entropy = torch.sum(prob * torch.log(prob), -1)
        return self.runner_params['entropy_coef'] * entropy.mean()
