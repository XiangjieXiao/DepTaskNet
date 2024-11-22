# import
import logging
from utils import create_logger, copy_all_src
from PPOMixRunner import MECRunner as Runner

##########################################################################################
# parameters
logger_params = {
    'folder_path': '../result2',
    'algorithm_name': 'DepTaskNet-Mix',
    'file_name': 'run_log',
}

env_params = {
    'mec_resource_params': {
        # for latency
        'mec_process_capable': (10.0 * 1024 * 1024),
        'mobile_process_capable': (1.0 * 1024 * 1024),
        'bandwidth_up': 6.0,
        'bandwidth_dl': 6.0,

        # for energy
        'rho': 1.25 * 10 ** -26,
        'f_l': 0.8 * 10 ** 9,
        'zeta': 3,
        'power_up': 1.258,
        'power_cloud': 0,
        'power_dl': 1.181,
    },

    'train_dataset': {
        # 'load_type': 'gv',
        'load_type': 'npy',
        'graph_number_per_file': 5000,
        'graph_file_path': [
            "../dataset/Origin dataset/task10/random.10.",
            "../dataset/Origin dataset/task15/random.15.",
            "../dataset/Origin dataset/task20/random.20.",
            "../dataset/Origin dataset/task25/random.25.",
            "../dataset/Origin dataset/task30/random.30.",
            "../dataset/Origin dataset/task35/random.35.",
            "../dataset/Origin dataset/task40/random.40.",
            "../dataset/Origin dataset/task45/random.45.",
            "../dataset/Origin dataset/task50/random.50.",
        ],
        'numpy_file_path': [
            '../dataset/Preprocess dataset/task10_6Mbps',
            '../dataset/Preprocess dataset/task15_6Mbps',
            '../dataset/Preprocess dataset/task20_6Mbps',
            '../dataset/Preprocess dataset/task25_6Mbps',
            '../dataset/Preprocess dataset/task30_6Mbps',
            '../dataset/Preprocess dataset/task35_6Mbps',
            '../dataset/Preprocess dataset/task40_6Mbps',
            '../dataset/Preprocess dataset/task45_6Mbps',
            '../dataset/Preprocess dataset/task50_6Mbps',
        ]
    },
    'eval_dataset': {
        # 'load_type': 'gv',
        'load_type': 'npy',
        'graph_number_per_file': 100,
        'graph_file_path': [
            "../dataset/Origin dataset/task10_test/random.10.",
            "../dataset/Origin dataset/task15_test/random.15.",
            "../dataset/Origin dataset/task20_test/random.20.",
            "../dataset/Origin dataset/task25_test/random.25.",
            "../dataset/Origin dataset/task30_test/random.30.",
            "../dataset/Origin dataset/task35_test/random.35.",
            "../dataset/Origin dataset/task40_test/random.40.",
            "../dataset/Origin dataset/task45_test/random.45.",
            "../dataset/Origin dataset/task50_test/random.50.",
        ],
        'numpy_file_path': [
            '../dataset/Preprocess dataset/task10_test_6Mbps',
            '../dataset/Preprocess dataset/task15_test_6Mbps',
            '../dataset/Preprocess dataset/task20_test_6Mbps',
            '../dataset/Preprocess dataset/task25_test_6Mbps',
            '../dataset/Preprocess dataset/task30_test_6Mbps',
            '../dataset/Preprocess dataset/task35_test_6Mbps',
            '../dataset/Preprocess dataset/task40_test_6Mbps',
            '../dataset/Preprocess dataset/task45_test_6Mbps',
            '../dataset/Preprocess dataset/task50_test_6Mbps',
        ]

    }
}

# encoder_hidden = decoder_hidden = hidden_size
model_params = {
    'seq2seq_params': {
        'encoder_params': {
            'input_feature': 17,
            'embedding_dim': 128,
            'encoder_hidden': 128,
            'bidirectional': True,
            'num_layers_for_one_LSTM': 1,
            'layer_norm': False,
        },
        'decoder_params': {
            'embedding_dim': 128,
            'decoder_hidden': 128,
            'num_layers': 2,
            'output_projection': 2,
            'layer_norm': False,
            'mlp_param': {
                'pref_input_dim': 2,
                'hidden_dim': 256,
            },
            'mid_embedding': 2,
        },
        # decode_type (default: greedy)
        'decode_type': 'random_sampling',
        # 'decode_type': 'epsilon_greedy',
        # 'epsilon': 0.01,
    },
    'critic_param': {
        'input_feature': 2,  # decoder_params: output_projection
        'hidden_layer1': 128,
        'hidden_layer2': 256,
        'hidden_layer3': 128,
        'output_projection': 2,
    },
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [50, 100],
        'gamma': 0.5,
    }
}

runner_params = {
    'use_cuda': True,
    'cuda_device_num': 0,
    'model_path': None,
    # for testing
    'skip_train': False,

    'epochs': 500,

    # for PPO epoch dataset init
    'pref_set': {
        'batch_size': 1024,
        'pref_flag': 'random',
        'pref_per_epoch': 5,
        # 'pref_flag': 'step',
        # 'pref_step': 0.1,                      # num_weight_per_epoch = size(0:pref_step:1)
    },

    # for PPO update
    'sample_reuse': 4,
    'clip_range': 0.2,
    'vf_coef': 0.5,
    'entropy_coef': 0.01,

    # TD-error & GAE
    'gamma': 0.99,          # discount
    'lambda': 0.95,         # GAE discount

    'eval_num': 100,
    'eval_pref_step': 0.1,

    'max_grad_clip': 1,
}


def print_config():
    logger = logging.getLogger('root')
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


def main():
    save_path = create_logger(**logger_params)
    print_config()
    copy_all_src(save_path)

    runner = Runner(env_params=env_params,
                    model_params=model_params,
                    optimizer_params=optimizer_params,
                    runner_params=runner_params,
                    logger_params=logger_params)
    runner.set_save_path(save_path)
    runner.run_ppo()


if __name__ == "__main__":
    main()
