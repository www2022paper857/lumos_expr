import os
import sys
import json
import dill
import copy
import pickle
import numpy as np

from enum import Enum, unique
from collections import defaultdict

from utils import *
from conf import LumosConf
from stat_encoder.fa import FAEncoder
from stat_encoder.pca import PCAEncoder
from stat_encoder.fft_stat import FFTStatEncoder


class RecordEntry(object):

    def __init__(self, inst_type, metrics, raw_metrics, jct, ts):
        # raw features
        self.inst_type = inst_type
        self.metrics = metrics
        self.raw_metrics = raw_metrics
        self.ts = ts
        # raw label
        self.jct = jct
        # rank label
        self.rank = -1


    def __repr__(self):
        repr_dict = {
            'inst_type': self.inst_type,
            'metrics.shape': self.metrics.shape,
            'ts': self.ts,
            'jct': self.jct,
            'rank': self.rank
        }
        import json
        return json.dumps(repr_dict, indent=4)


class DataLoaderOrdinal(object):

    def __init__(self, dump_pth=None, ordinal=True):
        conf = LumosConf()
        self.ds_root_pth = conf.get('dataset', 'path')
        self.vendor_cnt = conf.get('dataset', 'vendor_cnt')
        self.__data = None
        self.dump_pth = dump_pth
        # sampling interval
        self.sampling_interval = 5
        # the label is ordinal or raw
        self.ordinal = ordinal


    def load_data(self):
        '''
        old load_data, interval=5s
        '''
        self.sampling_interval = 5
        if self.dump_pth:
            self.__load_data_from_file()
            return

        self.__data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))

        def is_vendor(v):
            return '.' not in v

        for vendor in os.listdir(self.ds_root_pth):
            if not is_vendor(vendor): continue
            v_pth = os.path.join(self.ds_root_pth, vendor, '5 second')
            for inst_type in os.listdir(v_pth):
                i_pth = os.path.join(v_pth, inst_type)
                for w in os.listdir(i_pth):
                    [scale, rnd] = w.strip().split('_')[-2:]
                    if rnd not in ['1', '2', '3']: continue
                    workload = '_'.join(w.strip().split('_')[:2])
                    w_pth = os.path.join(i_pth, w)
                    repo_pth = os.path.join(w_pth, 'report.json')
                    metr_pth = os.path.join(w_pth, 'sar.csv')
                    [ts, jct] = mget_json_values(repo_pth, 'timestamp', 'elapsed_time')
                    ts = encode_timestamp(ts)
                    jct = float(jct)
                    header, metrics = read_csv(metr_pth)
                    if not header or not metrics: continue
                    norm_metrics = normalize_metrics(metrics, centralize=True)
                    raw_metrics = np.array(metrics)
                    self.__data[rnd][workload][scale].append(
                        RecordEntry(inst_type, norm_metrics, raw_metrics, jct, ts)
                    )


    def load_data_by_interval(self, interval=5):
        '''
        load data with specific sampling interval
        '''
        assert interval in (1, 5), 'invalid interval'
        self.sampling_interval = interval
        if interval == 5:
            self.load_data()
            return

        if self.dump_pth:
            self.__load_data_from_file()
            return

        self.__data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [])))

        def is_vendor(v):
            return '.' not in v

        for vendor in os.listdir(self.ds_root_pth):
            if not is_vendor(vendor): continue
            v_pth_1 = os.path.join(self.ds_root_pth, vendor)
            if vendor != 'aws':
                v_pth_1 = os.path.join(v_pth_1, '1 second')
            for inst_type in os.listdir(v_pth_1):
                i_pth = os.path.join(v_pth_1, inst_type)
                for w in os.listdir(i_pth):
                    # orderby_large can not be run successfully
                    if 'hive_orderby' in w: continue
                    w_after_split = w.strip().split('_')
                    scale, rnd = '', ''
                    # BigBench benchmarks only run for once
                    if len(w_after_split) == 4:
                        [scale, rnd] = w.strip().split('_')[-2:]
                        if rnd not in ['1', '2', '3']: continue
                    elif len(w_after_split) == 3:
                        rnd = '1'
                        scale = w_after_split[-1]
                    workload = '_'.join(w_after_split[:2])
                    w_pth = os.path.join(i_pth, w)
                    repo_pth = os.path.join(w_pth, 'report.json')
                    metr_pth = os.path.join(w_pth, 'sar.csv')
                    ts_key = ''
                    if len(w_after_split) == 4: ts_key = 'timestamp'
                    elif len(w_after_split) == 3: ts_key = 'begin_time'
                    [ts, jct] = mget_json_values(repo_pth, ts_key, 'elapsed_time')
                    ts = encode_timestamp(ts)
                    jct = float(jct)
                    header, metrics = read_csv(metr_pth)
                    if not header or not metrics: continue
                    norm_metrics = normalize_metrics(metrics, centralize=True)
                    raw_metrics = np.array(metrics)
                    self.__data[rnd][workload][scale].append(
                        RecordEntry(inst_type, norm_metrics, raw_metrics, jct, ts)
                    )
            if vendor == 'aws': continue
            v_pth_2 = os.path.join(self.ds_root_pth, vendor, '5 second')
            for inst_type in os.listdir(v_pth_2):
                i_pth = os.path.join(v_pth_2, inst_type)
                for w in os.listdir(i_pth):
                    [scale, rnd] = w.strip().split('_')[-2:]
                    if scale in ('tiny', 'small'): continue
                    if rnd not in ['1', '2', '3']: continue
                    workload = '_'.join(w.strip().split('_')[:2])
                    w_pth = os.path.join(i_pth, w)
                    repo_pth = os.path.join(w_pth, 'report.json')
                    metr_pth = os.path.join(w_pth, 'sar.csv')
                    [ts, jct] = mget_json_values(repo_pth, 'timestamp', 'elapsed_time')
                    ts = encode_timestamp(ts)
                    jct = float(jct)
                    header, metrics = read_csv(metr_pth)
                    if not header or not metrics: continue
                    norm_metrics = normalize_metrics(metrics, centralize=True)
                    raw_metrics = np.array(metrics)
                    self.__data[rnd][workload][scale].append(
                        RecordEntry(inst_type, norm_metrics, raw_metrics, jct, ts)
                    )


    def get_data(self):
        return self.__data


    def get_data_rankize(self):
        '''
        sort the performance of a workload running with a concrete input scale
        '''
        assert self.__data, 'data not loaded'
        rankize_data = copy.deepcopy(self.__data)
        for rnd, rnd_data in rankize_data.items():
            for wl, wl_data in rnd_data.items():
                for scale in wl_data:
                    scale_data = wl_data[scale]
                    sorted_scale_data = sorted(scale_data, key=lambda x: x.jct)
                    for record in sorted_scale_data:
                        record.rank = sorted_scale_data.index(record)
                    wl_data[scale] = sorted_scale_data

        return rankize_data


    def get_train_test_data(self, train_scale='tiny', test_wl='', flag='single'):
        '''
        get the training data that profiled on a concrete instance type
        param:
        @t_inst_type: the instance type that is used for profiling
        @test_wl: the workload that is to be used for testing
        '''
        rankize_data = self.get_data_rankize()
        assert test_wl in self.__data['1'] or test_wl in ('HiBench', 'BigBench'), 'invalid test workload'
        assert flag in ('single', 'multi'), 'indicating single/multi testing workloads'

        def is_test_wl(wl):
            if flag == 'single':
                return wl == test_wl
            else:
                if test_wl == 'BigBench':
                    return 'hive' in wl
                elif test_wl == 'HiBench':
                    return 'hive' not in wl

        conf = LumosConf()
        truncate = conf.get('dataset', 'truncate')
        fft_stat_encoder = FFTStatEncoder(truncate=truncate)

        train_data = defaultdict(lambda: defaultdict(lambda: {
            'X': [],
            'Y': []
        }))
        test_data = defaultdict(lambda: defaultdict(lambda: \
            defaultdict(lambda: defaultdict(lambda: {
            'X': [],
            'Y': []
        }))))

        predict_scales = ['tiny', 'small', 'large', 'huge']
        if train_scale == 'small':
            predict_scales.remove('tiny')

        for rnd, rnd_data in rankize_data.items():
            for wl, wl_data in rnd_data.items():
                if is_test_wl(wl): continue
                for record1 in wl_data[train_scale]:
                    t_inst_type = record1.inst_type
                    test_conf = conf.get_inst_detailed_conf(t_inst_type, format='list')
                    test_metrics_vec = fft_stat_encoder.encode(record1.metrics, record1.raw_metrics, sampling_interval=self.sampling_interval)
                    for scale in predict_scales:
                        target_scale = conf.get_scale_id(scale)
                        for record2 in wl_data[scale]:
                            target_conf = conf.get_inst_detailed_conf(record2.inst_type, format='list')
                            target_rank = record2.rank
                            target_jct = record2.jct
                            X = test_conf.copy()
                            X.extend(target_conf)
                            X.append(target_scale)
                            X.extend(test_metrics_vec)
                            train_data[rnd][t_inst_type]['X'].append(X)
                            if self.ordinal:
                                train_data[rnd][t_inst_type]['Y'].append(target_rank)
                            else:
                                train_data[rnd][t_inst_type]['Y'].append(target_jct)

        for rnd, rnd_data in rankize_data.items():
            for wl, wl_data in rnd_data.items():
                if not is_test_wl(wl): continue
                # wl_data = rnd_data[test_wl]
                for record1 in wl_data[train_scale]:
                    t_inst_type = record1.inst_type
                    test_conf = conf.get_inst_detailed_conf(t_inst_type, format='list')
                    test_metrics_vec = fft_stat_encoder.encode(record1.metrics, record1.raw_metrics, sampling_interval=self.sampling_interval)
                    for scale in predict_scales:
                        target_scale = conf.get_scale_id(scale)
                        for record2 in wl_data[scale]:
                            target_conf = conf.get_inst_detailed_conf(record2.inst_type, format='list')
                            target_rank = record2.rank
                            target_jct = record2.jct
                            X = test_conf.copy()
                            X.extend(target_conf)
                            X.append(target_scale)
                            X.extend(test_metrics_vec)
                            test_data[wl][rnd][t_inst_type][scale]['X'].append(X)
                            if self.ordinal:
                                test_data[wl][rnd][t_inst_type][scale]['Y'].append(target_rank)
                            else:
                                test_data[wl][rnd][t_inst_type][scale]['Y'].append(target_jct)

        return train_data, test_data


    @staticmethod
    def get_train_test_data_external(test_wl, train_scale, truncate=False, ordinal=False):
        assert train_scale == 'small', 'currently the model evaluated using small as the trianing scale'
        conf = LumosConf()
        dmp_pre = conf.get('dataset', 'train_test_dump_prefix')
        dmp_suf = 'o%d_t%d' % (ordinal, truncate)
        wl_pth = os.path.join(dmp_pre, '%s_%s.pkl' % (test_wl, dmp_suf))
        with open(wl_pth, 'rb') as fd:
            (train_data, test_data) = dill.load(fd)
            return train_data, test_data


    def __load_data_from_file(self):
        with open(self.dump_pth, 'rb') as fd:
            self.__data = dill.load(fd)


if __name__ == "__main__":
    conf = LumosConf()
    dump_pth = conf.get('dataset', 'dump_pth_ordinal_with_truc_v1')
    # dataloader = DataLoaderOrdinal()
    dataloader = DataLoaderOrdinal(dump_pth=dump_pth)
    dataloader.load_data_by_interval(interval=1)
    data = dataloader.get_data()
    # calculate global max values
    # global_max_vals = defaultdict(lambda : [])
    # global_max_names = defaultdict(lambda : [])
    # for rnd, rnd_data in data.items():
    #     max_names = [None for _ in range(62)]
    #     max_vals = [0 for _ in range(62)]
    #     for wl, wl_data in rnd_data.items():
    #         for scale, scale_data in wl_data.items():
    #             for record in scale_data:
    #                 metrics = record.raw_metrics
    #                 local_max_vals = np.max(metrics, axis=0)
    #                 for i in range(62):
    #                     if local_max_vals[i] > max_vals[i]:
    #                         max_vals[i] = local_max_vals[i]
    #                         max_names[i] = '%s_%s_%s' % (wl, scale, record.inst_type)
    #                     # max_vals[i] = max(max_vals[i], local_max_vals[i])
    #     global_max_vals[rnd] = max_vals
    #     global_max_names[rnd] = max_names
    # with open('conf/global_max_names.json', 'w') as fd:
    #     fd.write(json.dumps(dict(global_max_names), indent=4))
    # with open('conf/global_max_vals.json', 'w') as fd:
        # fd.write(json.dumps(dict(global_max_vals), indent=4))
    # with open(dump_pth, 'wb') as fd:
        # dill.dump(data, fd)
    print(len(data['1']))
    print(len(data['2']))
    print(len(data['3']))
    # train_data, test_data = dataloader.get_train_test_data(train_scale='small', test_wl='BigBench', flag='multi')
    # train_data, test_data = dataloader.get_train_test_data(test_wl='spark_pagerank')
    # ordinal=True, truncate=False
    # dataloader.ordinal = True
    # conf.runtime_set('dataset', 'truncate', False)
    # for wl in data['1'].keys():
    #     print('[ordinal=True, truncate=False] generating train/test data for workload %s...' % wl)
    #     train_data, test_data = dataloader.get_train_test_data(test_wl=wl, train_scale='small')
    #     with open(os.path.join(conf.get('dataset', 'train_test_dump_prefix'), '%s_o1_t0.pkl' % wl), 'wb') as fd:
    #         dill.dump((train_data, test_data), fd)

    # # ordinal=True, truncate=True
    # dataloader.ordinal = True
    # conf.runtime_set('dataset', 'truncate', True)
    # for wl in data['1'].keys():
    #     print('[ordinal=True, truncate=True] generating train/test data for workload %s...' % wl)
    #     train_data, test_data = dataloader.get_train_test_data(test_wl=wl, train_scale='small')
    #     with open(os.path.join(conf.get('dataset', 'train_test_dump_prefix'), '%s_o1_t1.pkl' % wl), 'wb') as fd:
    #         dill.dump((train_data, test_data), fd)

    # # ordinal=False, truncate=True
    # dataloader.ordinal = False
    # conf.runtime_set('dataset', 'truncate', True)
    # for wl in data['1'].keys():
    #     print('[ordinal=False, truncate=True] generating train/test data for workload %s...' % wl)
    #     train_data, test_data = dataloader.get_train_test_data(test_wl=wl, train_scale='small')
    #     with open(os.path.join(conf.get('dataset', 'train_test_dump_prefix'), '%s_o0_t1.pkl' % wl), 'wb') as fd:
    #         dill.dump((train_data, test_data), fd)

    # # ordinal=False, truncate=False
    # dataloader.ordinal = False
    # conf.runtime_set('dataset', 'truncate', False)
    # for wl in data['1'].keys():
    #     print('[ordinal=False, truncate=False] generating train/test data for workload %s...' % wl)
    #     train_data, test_data = dataloader.get_train_test_data(test_wl=wl, train_scale='small')
    #     with open(os.path.join(conf.get('dataset', 'train_test_dump_prefix'), '%s_o0_t0.pkl' % wl), 'wb') as fd:
    #         dill.dump((train_data, test_data), fd)
