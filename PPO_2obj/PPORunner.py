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
from PPOEnv import OffloadingEnvironment as Env
from PPOModel import ActorCritic


class PPOMemory:
    def __init__(self):
        self.data_map = None
        self.loss_map = None
        self._num_instance = None
        self._episode = None

    def push(self, data_map):
        if self.data_map is None:
            for key in data_map:
                data_map[key] = data_map[key]
            self.data_map = data_map
        else:
            for key in data_map:
                self.data_map[key] = np.concatenate((self.data_map[key], data_map[key]), axis=0)

    def push_loss(self, loss_map):
        if self.loss_map is None:
            for key in loss_map:
                loss_map[key] = loss_map[key].detach().cpu().numpy()
            self.loss_map = loss_map
        else:
            for key in loss_map:
                self.loss_map[key] = np.append(self.loss_map[key], loss_map[key].detach().cpu().numpy())

    def iterate_init(self, shuffle):
        if shuffle:
            perm = np.arange(self.data_map['problems'].shape[0])
            np.random.shuffle(perm)
            for key in self.data_map:
                self.data_map[key] = self.data_map[key][perm]

        self._num_instance = self.data_map['problems'].shape[0]
        self._episode = 0

    def get_batch(self, batch_size):
        remaining = self._num_instance - self._episode
        cur_batch_size = min(batch_size, remaining)

        batch = dict.fromkeys(self.data_map)
        for key in self.data_map:
            batch[key] = self.data_map[key][self._episode:self._episode + cur_batch_size]

        self._episode += cur_batch_size

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
        self.logger.info('--------------------------------------------------------------------------')
        self.env.init_epoch_dataset(**self.runner_params['pref_set'])
        self.env_eval = Env(train_flag=False, **self.env_params)
        self.logger.info('--------------------------------------------------------------------------')
        self.logger.info('Environment complete')

        self.memory = PPOMemory()

        self.actor_old = ActorCritic(**self.model_params)
        self.actor_new = ActorCritic(**self.model_params)
        self.optimizer = Optimizer(self.actor_new.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **self.optimizer_params['scheduler'])

        if runner_params['model_path'] is not None:
            self.actor_old.load_state_dict(torch.load(runner_params['model_path']))
            self.actor_new.load_state_dict(torch.load(runner_params['model_path']))

    def set_save_path(self, folder):
        self.save_path = folder

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
                self.sample()

                self.logger.info("sample time cost: {:.4f} s".format(time.time() - start_time))
                self.logger.info('Sample Reward: {:.4f}'.format(self.memory.data_map['rewards'].sum(-1).mean()))

                # update
                self.logger.info(fmt_row(13, ['total_loss', 'policy_loss', 'value_loss', 'policy_entropy']))
                start_time = time.time()
                for _ in range(self.runner_params['sample_reuse']):
                    self.memory.iterate_init(shuffle=False)
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
                self.logger.info("update time cost: {:.4f} s".format((time.time() - start_time)))

                self.log_data(dict(epoch=np.array(epoch),
                                   rewards=self.memory.data_map['rewards'].sum(-1).mean(),
                                   update_policy_loss=self.memory.loss_map['pg_loss'].mean(),
                                   update_value_loss=self.memory.loss_map['vf_loss'].mean(),
                                   update_entropy_loss=self.memory.loss_map['entropy'].mean()))

            # for eval & plot
            if (epoch == 1) | (epoch % 5 == 0):
                start_time = time.time()
                eval_log = self.eval()
                eval_log = dict(eval_pref=eval_log['eval_pref'],
                                eval_greedy_decode_result=eval_log['eval_greedy_decode_result'],
                                eval_greedy_decode_qoe=eval_log['eval_greedy_decode_qoe'],
                                eval_local_result=eval_log['eval_local_result'],
                                eval_remote_result=eval_log['eval_remote_result'])
                self.logger.info("Test in Eval dateset: {:.4f} s".format((time.time() - start_time)))
                self.logger.info('-------------------------------------------')
                self.logger.info(fmt_row(12, ['Class', 'Latency', 'Energy']))
                self.logger.info(fmt_row(12, ['1:0',
                                              eval_log['eval_greedy_decode_result'].sum(2).mean(1)[0, 0],
                                              eval_log['eval_greedy_decode_result'].sum(2).mean(1)[0, 1]]))
                self.logger.info(fmt_row(12, ['Local', eval_log['eval_local_result'].sum(1).mean(0)[0],
                                              eval_log['eval_local_result'].sum(1).mean(0)[1]]))
                self.logger.info(fmt_row(12, ['0.5:0.5',
                                              eval_log['eval_greedy_decode_result'].sum(2).mean(1)[len(eval_log['eval_greedy_decode_result']) // 2, 0],
                                              eval_log['eval_greedy_decode_result'].sum(2).mean(1)[len(eval_log['eval_greedy_decode_result']) // 2, 1]]))
                self.logger.info(fmt_row(12, ['0:1',
                                              eval_log['eval_greedy_decode_result'].sum(2).mean(1)[-1, 0],
                                              eval_log['eval_greedy_decode_result'].sum(2).mean(1)[-1, 1]]))
                self.logger.info(fmt_row(12, ['Remote', eval_log['eval_remote_result'].sum(1).mean(0)[0],
                                              eval_log['eval_remote_result'].sum(1).mean(0)[1]]))
                self.logger.info('-------------------------------------------')

                self.log_data(dict(eval_pref=np.expand_dims(eval_log['eval_pref'], axis=0),
                                   eval_greedy_decode_result=np.expand_dims(eval_log['eval_greedy_decode_result'],
                                                                            axis=0),
                                   eval_local_result=np.expand_dims(eval_log['eval_local_result'], axis=0),
                                   eval_remote_result=np.expand_dims(eval_log['eval_remote_result'], axis=0)))
                self.logger.info('Eval Reward: {:.4f}'.format(eval_log['eval_greedy_decode_qoe'].mean()))

            torch.save(self.actor_old.state_dict(), self.save_path + 'model.pth')
            np.save(self.save_path + 'data_log.npy', self.data_log)
            self.logger.info('-------------------------------------------')

    def sample(self):
        self.actor_old.set_decode_type(self.actor_old.model_params['seq2seq_params']['decode_type'])
        self.env.init_idx(shuffle=True, train_set=self.runner_params['pref_set'])
        while len(self.env.batch_idx) > 0:
            self.env.load_batch(self.runner_params['pref_set']['batch_size'])
            data = torch.tensor(self.env.problems, dtype=torch.float32)
            pref = torch.tensor(self.env.pref, dtype=torch.float32)

            probs, actions, logits, value = self.actor_old(data, pref)

            self.env.get_reward(copy.deepcopy(actions))
            rewards = (self.env.pref * self.env.rewards).sum(-1)

            advantages, tdlamret, _ = self.get_GAE(torch.tensor(rewards, dtype=torch.float32),
                                                   value)

            self.memory.push(dict(problems=self.env.problems,
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
        eval_result = []
        eval_reward = []

        pref_list = np.arange(0, 1 + 1e-10, self.runner_params['eval_pref_step'])
        pref_list = np.vstack([np.ones(pref_list.shape) - pref_list, pref_list]).T
        for pref in pref_list:
            self.env_eval.init_idx(shuffle=False)
            self.env_eval.load_batch(self.runner_params['eval_num'])
            data = torch.tensor(self.env_eval.problems, dtype=torch.float32)

            self.env_eval.init_batch_reward()
            self.actor_old.set_decode_type('greedy')
            # self.actor_old.set_decode_type('multinomial')
            _, actions, _, _ = self.actor_old(data, torch.tensor(pref, dtype=torch.float32))
            self.env_eval.get_reward(copy.deepcopy(actions), reward_type='qoe')

            eval_result.append(np.stack([self.env_eval.latency, self.env_eval.energy], axis=-1))
            eval_reward.append((pref * self.env_eval.rewards).sum(-1).mean())

        eval_result = np.array(eval_result)
        eval_reward = np.array(eval_reward)
        return dict(eval_pref=pref_list,
                    eval_greedy_decode_result=eval_result,
                    eval_greedy_decode_qoe=eval_reward,
                    eval_local_result=self.env_eval.baseline['local_le'],
                    eval_remote_result=self.env_eval.baseline['remote_le'])

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

    def log_data(self, epoch_data):
        for key in epoch_data:
            if key in self.data_log:
                self.data_log[key] = np.vstack((self.data_log[key], epoch_data[key]))
            else:
                self.data_log[key] = epoch_data[key]
