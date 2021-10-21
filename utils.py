import json
import time
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import normalize


class Timer(object):
    def __init__(self):
        self.ts = -1
        self.started = False
        self.__elapsed_time = 0


    def start(self):
        self.ts = time.time()
        self.started = True


    def stop(self):
        assert(self.started, 'timer not started')
        self.__elapsed_time = time.time() - self.ts
        self.started = False


    def get_elasped_time(self):
        return self.__elapsed_time


def singleton(cls):
    _instance = {}

    def inner():
        if cls not in _instance:
            _instance[cls] = cls()
        return _instance[cls]

    return inner


def read_csv(csv_file_path):
    with open(csv_file_path, 'r') as fd:
        lines = fd.readlines()
        header, content = lines[0], lines[1:]
        content = [line.strip().split(',')[1:] for line in content] # remove ts in the first column
        for i in range(len(content)):
            for j in range(len(content[i])):
                ele = content[i][j]
                if ele == 'lo': # IFACE
                    content[i][j] = 0
                elif ele.replace('.', '').isdigit():
                    content[i][j] = float(ele)
                else:
                    return None, None

        col_num_set = set([len(line) for line in content])
        if len(col_num_set) != 1:
            print('bad csv file: %s, lines: %s' % (csv_file_path, col_num_set))
            return None, None
        return header, content


def get_json_value(json_file_path, *keys):
    with open(json_file_path, 'r') as fd:
        data = json.load(fd)
        tmp = data[keys[0]]
        if len(keys) > 1:
            for i in range(len(keys) - 1):
                tmp = tmp[keys[i + 1]]
        return tmp


def mget_json_values(json_file_path, *key_arr):
    with open(json_file_path, 'r') as fd:
        data = json.load(fd)
        ret = []
        for keys in key_arr:
            if isinstance(keys, list):
                tmp = data[keys[0]]
                if len(keys) > 1:
                    for i in range(len(keys) - 1):
                        tmp = tmp[keys[i + 1]]
                    ret.append(tmp)
            elif isinstance(keys, str):
                ret.append(data[keys])
        return ret


def encode_timestamp(ts):
    '''
    encoded features: 1) day of the weak; 2) hour of the day
    '''
    f_time = time.strptime(ts, '%Y-%m-%d %H:%M:%S')
    day_code = f_time.tm_wday / 7
    hour_code = f_time.tm_hour / 24
    return [day_code, hour_code]


def normalize_metrics(metrics, centralize=True):
    '''
    normalize the metrics data for each feature
    '''
    norm_metrics = np.array(metrics)
    norm_metrics = normalize(norm_metrics, axis=0)
    if centralize:
        norm_metrics -= np.mean(norm_metrics, axis=0)
    return norm_metrics


def get_max_lens(data):
    max_lens = defaultdict(lambda: 0)
    for wl, _data in data.items():
        max_lens[wl] = max(max_lens[wl], max(ele.get_metrics().shape[0] for ele in _data))
    return max_lens


def get_indices(tol, part):
    '''
    select #part (20% by default) indices from [0, tol - 1]
    '''
    seed = 20200924 % 1639 % tol
    gap = tol // part
    tol_indices = list(range(tol))
    indices = [(seed + gap * i) % tol for i in range(part)]
    return indices


def get_left_indices(tol, part):
    '''
    get the left 80% indices
    '''
    indices = get_indices(tol, part)
    tol_indices = list(range(tol))
    left_indices = set(tol_indices) - set(indices)
    return left_indices


def get_samples(data):
    '''
    sample from the original data, about 20% of the original data for encoder
    training
    '''
    samples = defaultdict(lambda: [])
    for wl, w_data in data.items():
        tol_size = sum(len(v_data) for vendor, v_data in w_data.items())
        for vendor, v_data in w_data.items():
            to_sample_cnt = max(int(len(v_data) * 0.2), 1)
            to_sample_idxes = get_indices(len(v_data), to_sample_cnt)
            for idx in to_sample_idxes:
                samples[wl].append(v_data[idx])
            
    return samples


def get_left_samples(data):
    '''
    samples left are used to train the prediction model
    '''
    samples = defaultdict(lambda: [])
    for wl, w_data in data.items():
        tol_size = sum(len(v_data) for vendor, v_data in w_data.items())
        for vendor, v_data in w_data.items():
            to_sample_cnt = max(int(len(v_data) * 0.2), 1)
            to_sample_idxes = get_left_indices(len(v_data), to_sample_cnt)
            for idx in to_sample_idxes:
                samples[wl].append(v_data[idx])
            
    return samples


def padding_data(data, max_len=None):
    _max_len = max_len if max_len else max(_data.get_metrics().shape[0] for _data in data)
    for i in range(len(data)):
        _data = data[i].get_metrics()
        if _data.shape[0] < max_len:
            post_padding = np.zeros((_max_len - _data.shape[0], _data.shape[1]))
            data[i].update_metrics(
                np.concatenate((_data, post_padding)), tag='pad')
