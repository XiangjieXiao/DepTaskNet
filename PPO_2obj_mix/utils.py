"""
The MIT License

Copyright (c) 2021 Yeong-Dae Kwon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.



THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import sys
import os
from datetime import datetime
import logging
import logging.config
import numpy as np
import matplotlib.pyplot as plt
import shutil


def create_logger(**params):
    folder_path = params['folder_path']+'/'+datetime.now().strftime("%Y%m%d_%H%M%S")+'_'+params['algorithm_name']+'/'

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if 'file_name' in params:
        filename = folder_path + params['file_name'] + '.txt'
    else:
        filename = folder_path + 'log.txt'

    file_mode = 'a' if os.path.isfile(filename) else 'w'

    root_logger = logging.getLogger()
    root_logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(filename)s(%(lineno)d) : %(message)s", "%Y-%m-%d %H:%M:%S")

    for hdlr in root_logger.handlers[:]:
        root_logger.removeHandler(hdlr)

    # write to file
    fileout = logging.FileHandler(filename, mode=file_mode)
    fileout.setLevel(logging.INFO)
    fileout.setFormatter(formatter)
    root_logger.addHandler(fileout)

    # write to console
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    root_logger.addHandler(console)

    return folder_path


def copy_all_src(dst_root):
    # execution dir
    if os.path.basename(sys.argv[0]).startswith('ipykernel_launcher'):
        execution_path = os.getcwd()
    else:
        execution_path = os.path.dirname(sys.argv[0])

    # home dir setting
    tmp_dir1 = os.path.abspath(os.path.join(execution_path, sys.path[0]))
    tmp_dir2 = os.path.abspath(os.path.join(execution_path, sys.path[1]))

    if len(tmp_dir1) > len(tmp_dir2) and os.path.exists(tmp_dir2):
        home_dir = tmp_dir2
    else:
        home_dir = tmp_dir1

    # make target directory
    dst_path = os.path.join(dst_root, 'src')

    if not os.path.exists(dst_path):
        os.makedirs(dst_path)

    main_file_copied = False

    for item in sys.modules.items():
        key, value = item

        if hasattr(value, '__file__') and value.__file__:
            src_abspath = os.path.abspath(value.__file__)

            if src_abspath == os.path.abspath(sys.argv[0]) and main_file_copied:
                continue
            elif src_abspath == os.path.abspath(sys.argv[0]):
                main_file_copied = True

            if os.path.commonprefix([home_dir, src_abspath]) == home_dir:
                dst_filepath = os.path.join(dst_path, os.path.basename(src_abspath))

                if os.path.exists(dst_filepath):
                    split = list(os.path.splitext(dst_filepath))
                    split.insert(1, '({})')
                    filepath = ''.join(split)
                    post_index = 0

                    while os.path.exists(filepath.format(post_index)):
                        post_index += 1

                    dst_filepath = filepath.format(post_index)

                shutil.copy(src_abspath, dst_filepath)


###################################################################
# for plot
###################################################################
def plot_epoch_obj(data_map):
    epoch = data_map['epoch']
    plt.figure(figsize=(9, 3), dpi=400)

    plt.subplot(131)
    plt.plot(epoch, data_map['loss'], label='loss', c='r')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss')
    plt.legend()

    plt.subplot(132)
    plt.plot(epoch, data_map['train_local_latency'], label='local', c='r')
    plt.plot(epoch, data_map['train_sample_decode_latency'], label='sample_decode', c='b')
    plt.plot(epoch, data_map['train_greedy_decode_latency'], label='greedy_decode', c='m')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Latency', fontsize=12)
    plt.title('Train Dataset Result')
    plt.legend(prop={'size': 6})

    plt.subplot(133)
    plt.plot(epoch, data_map['eval_local_latency'], label='local', c='r')
    plt.plot(epoch, data_map['eval_sample_decode_latency'], label='sample_decode', c='b')
    plt.plot(epoch, data_map['eval_greedy_decode_latency'], label='greedy_decode', c='m')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Latency', fontsize=12)
    plt.title('Eval Dataset Result')
    plt.legend(prop={'size': 6})

    plt.tight_layout()
    plt.show()


def fmt_row(width, row, header=False):
    out = " | ".join(fmt_item(x, width) for x in row)
    if header:
        out = out + "\n" + "-" * len(out)
    return out


def fmt_item(x, l):
    if isinstance(x, np.ndarray):
        assert x.ndim == 0
        x = x.item()
    if isinstance(x, (float, np.float32, np.float64)):
        v = abs(x)
        if (v < 1e-4 or v > 1e+4) and v > 0:
            rep = "%7.2e" % x
        else:
            rep = "%7.5f" % x
    else:
        rep = str(x)
    return " " * (l - len(rep)) + rep
