import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = Encoder(**self.model_params['seq2seq_params']['encoder_params'])
        if model_params['seq2seq_params']['encoder_params']['bidirectional']:
            encoder_output_dim = model_params['seq2seq_params']['encoder_params']['encoder_hidden'] * 2
        else:
            encoder_output_dim = model_params['seq2seq_params']['encoder_params']['encoder_hidden']
        self.decoder = Decoder(encoder_output_dim, **self.model_params['seq2seq_params']['decoder_params'])
        self.critic = Critic(**self.model_params['critic_param'])
        self.decode_type = model_params['seq2seq_params']['decode_type']
        self.encoded_nodes = None
        self.batch_size = None
        self.task_length = None
        self.decoder_input = None

    def forward(self, data, pref):
        self.batch_size, self.task_length, _ = data.shape

        # actor-encode
        self.encoded_nodes, (encoder_h, encoder_c) = self.encoder(data)

        # actor-decode
        state = self.decoder.initial_decoder(self.encoded_nodes, pref)
        probs, actions, decoder_output = None, None, None
        for step in range(self.task_length):
            rnn_output, state = self.decoder(state)
            prob = torch.softmax(rnn_output, dim=-1)

            if self.decode_type == 'teach_forcing':
                action = self.decoder_input[:, step:step+1]
            elif self.decode_type == 'epsilon_greedy':
                idx = torch.rand(self.batch_size) <= self.model_params['seq2seq_params']['epsilon']
                action = torch.argmax(prob, dim=-1)
                rand_decision = torch.randint(low=0, high=2, size=action.shape)
                action[idx] = rand_decision[idx]
            elif self.decode_type == 'random_sampling':
                action = prob.squeeze(dim=1).multinomial(num_samples=1)
            else:
                action = torch.argmax(prob, dim=-1)

            state[0] = action

            if probs is None or actions is None or decoder_output is None:
                probs = prob
                actions = action
                decoder_output = rnn_output
            else:
                probs = torch.cat([probs, prob], dim=1)
                actions = torch.cat([actions, action], dim=1)
                decoder_output = torch.cat([decoder_output, rnn_output], dim=1)

        value = self.critic(decoder_output)

        return probs, actions, decoder_output, value

    def set_decode_type(self, mode):
        self.decode_type = mode


#######################################
# Encoder
#######################################
class Encoder(nn.Module):
    def __init__(self, **encoder_params):
        super().__init__()
        self.encoder_params = encoder_params
        self.encoder_embedding = nn.Linear(in_features=encoder_params['input_feature'],
                                           out_features=encoder_params['embedding_dim'],
                                           bias=True)
        self.bi_lstm = nn.LSTM(input_size=encoder_params['embedding_dim'],
                               hidden_size=encoder_params['encoder_hidden'],
                               num_layers=encoder_params['num_layers_for_one_LSTM'],
                               bidirectional=encoder_params['bidirectional'],
                               batch_first=True)

    def forward(self, task_sequence):
        # [batch_size, task_num, embedding_dim]
        task_embedding = self.encoder_embedding(task_sequence)
        '''
        output [batch_size, task_num, num_directions * hidden_size]
              = concat([batch_size, task_num, :hidden_size],
                        batch_size, reverse_task_num, hidden_size:])
        h or c [num_layers * num_directions, batch_size, hidden_size]
        '''
        # output, (encoder_h, encoder_c) = self.bi_lstm(F.relu(task_embedding))
        output, (encoder_h, encoder_c) = self.bi_lstm(task_embedding)

        return output, (encoder_h, encoder_c)


#######################################
# Decoder
#######################################
class Decoder(nn.Module):
    def __init__(self, encoder_out_dim, **decoder_params):
        super().__init__()
        self.decoder_params = decoder_params
        self.encoder_out_dim = encoder_out_dim
        self.mid_embedding_dim = decoder_params['mlp_param']['pref_input_dim']
        self.decoder_embedding = nn.Embedding(num_embeddings=decoder_params['output_projection'],
                                              embedding_dim=decoder_params['embedding_dim'])

        self.lstm = nn.LSTM(input_size=decoder_params['embedding_dim'],
                            hidden_size=decoder_params['decoder_hidden'],
                            num_layers=decoder_params['num_layers'],
                            batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(in_features=decoder_params['mlp_param']['pref_input_dim'],
                      out_features=decoder_params['mlp_param']['hidden_dim'],
                      bias=True),
            # nn.ReLU(),
            nn.Linear(in_features=decoder_params['mlp_param']['hidden_dim'],
                      out_features=decoder_params['mlp_param']['hidden_dim'],
                      bias=True),
            # nn.ReLU(),
            nn.Linear(in_features=decoder_params['mlp_param']['hidden_dim'],
                      out_features=3*self.mid_embedding_dim,
                      bias=True),
        )

        # Preference query key value for attention
        self.projection_Wq = nn.Linear(in_features=self.mid_embedding_dim,
                                       out_features=decoder_params['decoder_hidden']*decoder_params['decoder_hidden'],
                                       bias=False)
        self.projection_Wk = nn.Linear(in_features=self.mid_embedding_dim,
                                       out_features=encoder_out_dim*self.decoder_params['decoder_hidden'],
                                       bias=False)
        self.projection_Wv = nn.Linear(in_features=self.mid_embedding_dim,
                                       out_features=encoder_out_dim * encoder_out_dim,
                                       bias=False)

        self.Wc = nn.Linear(in_features=encoder_out_dim+decoder_params['decoder_hidden'],
                            out_features=decoder_params['decoder_hidden'],
                            bias=False)

        self.output_layer = nn.Linear(in_features=decoder_params['decoder_hidden'],
                                      out_features=decoder_params['output_projection'],
                                      bias=False)

        self.Wq_pref = None
        self.att_value = None
        self.att_key = None
        self.batch_size = None
        self.task_length = None

    def initial_decoder(self, enc_outputs, pref):
        self.batch_size = enc_outputs.shape[0]

        mid_embedding = self.mlp(pref)

        # [batch, task_num, decoder_hidden]  for attention computing
        self.Wq_pref = self.projection_Wq(mid_embedding[:self.mid_embedding_dim])\
            .reshape(self.decoder_params['decoder_hidden'], self.decoder_params['decoder_hidden'])

        Wk_pref = self.projection_Wk(mid_embedding[self.mid_embedding_dim:2 * self.mid_embedding_dim])\
            .reshape(self.decoder_params['decoder_hidden'], self.encoder_out_dim)
        self.att_key = F.linear(enc_outputs, Wk_pref)

        # [batch, task_num, encoder_hidden * num_directions]
        Wv_pref = self.projection_Wv(mid_embedding[2 * self.mid_embedding_dim:3 * self.mid_embedding_dim])\
            .reshape(self.encoder_out_dim, self.encoder_out_dim)
        self.att_value = F.linear(enc_outputs, Wv_pref)

        # initial first decoder input & decoder lstm state
        # [step 1, batch, encoder_hidden * num_directions]
        start_token = torch.zeros([self.batch_size, 1], dtype=torch.long)
        # [num_layers, batch, decoder_hidden]
        h = torch.zeros([self.decoder_params['num_layers'],
                         self.batch_size,
                         self.decoder_params['decoder_hidden']], dtype=torch.float32)
        c = torch.zeros([self.decoder_params['num_layers'],
                         self.batch_size,
                         self.decoder_params['decoder_hidden']], dtype=torch.float32)

        state = [start_token, h, c]
        return state

    def luong_attention_score(self, hidden):
        # h_t^t * hs
        # [batch, 1, task_num]
        score = torch.bmm(hidden, self.att_key.transpose(1, 2))
        attention_score = torch.softmax(score, dim=2)

        return attention_score

    def forward(self, state):

        output, h, c = state
        output = self.decoder_embedding(output)
        output, (h, c) = self.lstm(output, (h, c))

        # weight of one node 2 all node
        # [batch, 1, task_num]
        att_q = F.linear(output, self.Wq_pref)
        alignments = self.luong_attention_score(att_q)

        # [batch, step 1, num_layers*decoder_hidden]
        attention_context = torch.bmm(alignments, self.att_value)
        attention_h = torch.tanh(self.Wc(torch.cat([attention_context, output], dim=2)))

        # [batch, step 1, offloading_decision]
        rnn_output = self.output_layer(attention_h)
        state = [attention_h, h, c]

        return rnn_output, state


class Critic(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.input_layer = nn.Linear(self.model_params['input_feature'],
                                     self.model_params['hidden_layer1'],
                                     bias=True)
        self.hidden_layer1 = nn.Linear(self.model_params['hidden_layer1'],
                                       self.model_params['hidden_layer2'],
                                       bias=True)
        self.hidden_layer2 = nn.Linear(self.model_params['hidden_layer2'],
                                       self.model_params['hidden_layer3'],
                                       bias=True)
        self.output_layer = nn.Linear(self.model_params['hidden_layer3'],
                                      self.model_params['output_projection'],
                                      bias=True)

    def forward(self, rnn_output):
        q1 = self.input_layer(rnn_output)
        q1_temp = F.relu(q1)
        q2 = self.hidden_layer1(q1_temp)
        q2_temp = F.relu(q2)
        q3 = self.hidden_layer2(q2_temp)
        q3_temp = F.relu(q3)
        q = self.output_layer(q3_temp)

        pi = F.softmax(rnn_output, dim=-1)
        value_function = pi * q
        value = value_function.sum(dim=-1)

        return value
