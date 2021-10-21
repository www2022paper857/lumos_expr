import pywt
import numpy as np
from conf import LumosConf
from utils import singleton
from scipy.fftpack import fft, ifft

class FFTStatEncoder(object):

    def __init__(self, truncate=True):
        self.truncate = truncate


    def encode(self, norm_data, raw_data, sampling_interval=5):
        ret = []
        conf = LumosConf()
        valid_idx = conf.get('dataset', 'selected_idx')
        if norm_data.shape[1] != len(valid_idx):
            norm_data = norm_data[:, valid_idx]
        if raw_data.shape[1] != len(valid_idx):
            raw_data = raw_data[:, valid_idx]
        for i in range(norm_data.shape[1]):
            tmp = []
            norm_series = norm_data[:, i]
            raw_series = raw_data[:, i]
            fft_feat = self.__fft(norm_series, sampling_interval=sampling_interval)
            stat_feat = self.__stat(raw_series, i)
            tmp.extend(fft_feat)
            tmp.extend(stat_feat)
            ret.extend(tmp)
        return ret


    def __fft(self, norm_series, n_feat=2, sampling_interval=5):
        if self.truncate:
            left, right = self.__key_stage_detect(norm_series)
            norm_series = norm_series[left: right]
        len_s = len(norm_series)
        N = int(np.power(2, np.ceil(np.log2(len_s))))
        fft_y = fft(norm_series, N)[:N // 2] / len_s * 2
        fft_y_abs = np.abs(fft_y)
        fre = np.arange(N // 2) / N * sampling_interval
        top_amp = np.sort(fft_y_abs)[-n_feat:]
        top_idx = np.argsort(fft_y_abs)[-n_feat:]
        top_fre = fre[top_idx]
        return list(top_amp) + list(top_fre)


    def __stat(self, raw_series, idx):
        conf = LumosConf()
        selected_idxes = conf.get('dataset', 'selected_idx')
        valid_max_val = conf.get_global_max_val(selected_idxes[idx])
        max_val = np.max(raw_series) / valid_max_val
        min_val = np.min(raw_series) / valid_max_val
        avg_val = np.mean(raw_series) / valid_max_val
        # var_val = np.var(raw_series) / valid_max_val
        return [max_val, min_val, avg_val]

    
    def __key_stage_detect(self, series_data, n_iter=3, adjacant_threshold=0.7):
        '''
        identify the key stage of a multi-stage recursive/iterative
        time series using haar wavelet transformation
        '''
        cA = series_data
        cA_arr = []
        for _ in range(n_iter):
            cA, cD = pywt.dwt(cA, 'haar')
        factor = int(2 ** n_iter)
        max_idx = np.argmax(cA)
        if cA[max_idx] == 0: return 0, len(series_data)
        left = max(max_idx - 1, 0)
        right = min(max_idx + 1, len(cA))
        while left > 0 and cA[left - 1] / cA[max_idx] > adjacant_threshold: left -= 1
        while right < len(cA) and cA[right] / cA[max_idx] > adjacant_threshold: right += 1
        return left * 2 ** n_iter, min(len(series_data), right * 2 ** n_iter)
