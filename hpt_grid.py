import os
import sys
import json
import dill
import copy
import argparse
import numpy as np

from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor

from utils import *
from conf import LumosConf
from data_loader_ordinal import DataLoaderOrdinal


def cal_top_3_acc(results):
    cnt = defaultdict(lambda: defaultdict(lambda: 0))
    for wl, wl_data in results.items():
        for scale, scale_data in wl_data.items():
            cnt[scale]['total'] += 1
            if np.argsort(scale_data['test_Y_bar'])[0] < 3:
                cnt[scale]['top_3'] += 1
    return cnt['large']['top_3'] / cnt['large']['total']


def cal_err(results, rank_data):
    err = defaultdict(lambda: {
        'abs_err': [],
        'rel_err': []
    })
    for wl, wl_data in results.items():
        for scale, scale_data in wl_data.items():
            optimal_bar_idx = np.argsort(scale_data['test_Y_bar'])[0]
            # optimal_bar = scale_data['test_Y'][optimal_bar_idx]
            optimal_bar = rank_data['1'][wl][scale][optimal_bar_idx].jct
            # optimal = scale_data['test_Y'][0]
            optimal = rank_data['1'][wl][scale][0].jct
            abs_err = optimal_bar - optimal
            rel_err = abs_err / optimal
            err[scale]['abs_err'] = abs_err
            err[scale]['rel_err'] = rel_err
    avg_abs_err = np.mean(err['large']['abs_err'])
    avg_rel_err = np.mean(err['large']['rel_err'])
    return avg_abs_err, avg_rel_err


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='grid search')
    parser.add_argument('-j', '--n_jobs', help='number of jobs running parallel', type=int, default=None)
    args = parser.parse_args()

    conf = LumosConf()
    dump_pth = conf.get('dataset', 'dump_pth_ordinal_with_truc_v1')
    # dataloader = DataLoaderOrdinal()
    dataloader = DataLoaderOrdinal(dump_pth=dump_pth)
    dataloader.load_data_by_interval(interval=1)
    rank_data = dataloader.get_data_rankize()

    # dataset options
    op_truncate = [True, False]
    op_ordinal = [True, False]
    # model options
    op_max_depth = [3, 4, 5]
    op_n_estimators = [10, 40, 70, 100]
    op_criterion = ['mse', 'mae']
    op_max_features = ['auto', 'sqrt', 'log2', 0.5]
    
    conf = LumosConf()
    dump_pth = conf.get('dataset', 'dump_pth_ordinal_with_truc_v1')
    dataloader = DataLoaderOrdinal(dump_pth=dump_pth)
    dataloader.load_data_by_interval(interval=1)
    data = dataloader.get_data()
    workloads = data['1'].keys()
    # workloads = list(data['1'].keys())[:3]

    best_conf = ''

    # making the model options static, tuning the dataset options
    # making the testing instance type and round static, tuning in the end
    model_init_ops = {
        'op_d': 3,
        'op_e': 10,
        'op_c': 'mse',
        'op_f': 'auto'
    }
    best_op_t, best_op_o = None, None
    best_top_3_acc = .0
    best_abs_err = 0x7ffffff
    best_rel_err = 1.0
    init_rnd, init_t_inst_type = '1', 'g6.large'
    for op_t in op_truncate:
        for op_o in op_ordinal:
            print('truncate=%s, ordinal=%s' % (op_t, op_o))
            dmp_pre = conf.get('dataset', 'train_test_dump_prefix')
            dmp_suf = 'o%d_t%d' % (op_o, op_t)
            results = defaultdict(lambda: {})
            wl_cnt = 1
            for wl in workloads:
                print('processing workload #%d/%d' % (wl_cnt, len(workloads)), end='\r')
                wl_cnt += 1
                wl_pth = os.path.join(dmp_pre, '%s_%s.pkl' % (wl, dmp_suf))
                train_data, test_data = None, None
                with open(wl_pth, 'rb') as fd:
                    (train_data, test_data) = dill.load(fd)
                train_X = train_data[init_rnd][init_t_inst_type]['X']
                train_Y = train_data[init_rnd][init_t_inst_type]['Y']
                regressor = RandomForestRegressor(
                    n_estimators=model_init_ops['op_e'],
                    criterion=model_init_ops['op_c'],
                    max_depth=model_init_ops['op_d'],
                    max_features=model_init_ops['op_f'],
                    n_jobs=args.n_jobs
                )
                regressor.fit(train_X, train_Y)
                for scale, test_XY in test_data[init_rnd][init_t_inst_type].items():
                    test_X, test_Y = test_XY['X'], test_XY['Y']
                    test_Y_bar = regressor.predict(test_X)
                    results[wl][scale] = {
                        'test_Y_bar': test_Y_bar,
                        'test_Y': test_Y
                    }
            print()
            top_3_acc = cal_top_3_acc(results)
            abs_err, rel_err = cal_err(results, rank_data) 
            print('top_3_acc: %.2f%%, abs_err: %.2f, rel_err: %.2f%%' % (top_3_acc * 100, abs_err, rel_err * 100))
            if top_3_acc > best_top_3_acc:
                best_op_o = op_o
                best_op_t = op_t
                best_top_3_acc = top_3_acc
                best_abs_err = abs_err
                best_rel_err = rel_err
            elif top_3_acc == best_top_3_acc:
                if rel_err < best_rel_err:
                    best_op_o = op_o
                    best_op_t = op_t
                    best_abs_err = abs_err
                    best_rel_err = rel_err
    print('best_top_3_acc: %.2f%%, best_abs_err: %.2f, best_rel_err: %.2f%%' % (best_top_3_acc * 100, best_abs_err, best_rel_err * 100))
    print('best conf: truncate=%s, ordinal=%s' % (best_op_o, best_op_t))
