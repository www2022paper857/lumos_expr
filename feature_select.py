import json
import random
import numpy as np
from sklearn.cluster import KMeans
from data_loader_ordinal import DataLoaderOrdinal


def cal_std_and_cof(metrics):
    return np.std(metrics, axis=0), np.corrcoef(metrics.T)


def ana_metrics(metrics_data):
    stat_res = []
    for metrics in metrics_data:
        std, cof = cal_std_and_cof(metrics)
        stat_res.append((std, cof))
    return stat_res


def select_features(metrics_data, cof_threshold=0.6):
    '''
    select key features and return their indexes
    '''
    stat_res = []
    for metrics in metrics_data:
        std, cof = cal_std_and_cof(metrics)
        stat_res.append((std, cof))
    std_data = [[] for _ in range(np.shape(stat_res[0][0])[0])]
    for res in stat_res:
        for i in range(len(res[0])):
            std_data[i].append(res[0][i])

    # step 1: select features with larger variances
    std_medians = np.median(std_data, axis=1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit([[x, 0] for x in std_medians])
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    valid_label = 0 if cluster_centers[0][0] > cluster_centers[1][0] else 1
    valid_idxes = np.where(labels == valid_label)[0]
    print(valid_idxes)

    # step 2: remove redundant features having high cof
    avg_cof = np.zeros(np.shape(stat_res[0][1]))
    for res in stat_res:
        avg_cof += np.nan_to_num(res[1])
    avg_cof /= len(stat_res)
    
    selected_feat_idxes = []
    redundant_feat_idxes = []
    for idx in range(len(avg_cof)):
        if idx not in valid_idxes: continue
        if idx in redundant_feat_idxes: continue
        selected_feat_idxes.append(idx)
        for idx_2 in range(len(avg_cof[idx])):
            if idx_2 in selected_feat_idxes: continue
            if idx_2 in redundant_feat_idxes: continue
            if avg_cof[idx][idx_2] > cof_threshold and idx_2 != idx:
                redundant_feat_idxes.append(idx_2)
    
    return selected_feat_idxes


if __name__ == "__main__":
    from conf import LumosConf
    conf = LumosConf()
    dump_pth = conf.get('dataset', 'dump_pth_ordinal_with_truc_v1')
    dataloader = DataLoaderOrdinal(dump_pth=dump_pth)
    dataloader.load_data_by_interval(interval=1)
    data = dataloader.get_data()
    to_select_scale = 'large'
    metrics_data = []
    for wl, wl_data in data['1'].items():
        scale_data = wl_data[to_select_scale]
        # metrics_data.append(random.sample(scale_data, 1)[0].metrics)
        metrics_data.append(scale_data[1].metrics)
    feature_idxes = select_features(metrics_data)
    # ana_metrics(metrics_data)
    print('%d features selected: %s' % (len(feature_idxes), feature_idxes))
