import os
import json
from utils import singleton


@singleton
class LumosConf(object):

    def __init__(self):
        with open('conf/lumos.json') as fd:
            self.__conf = json.load(fd)
        with open('conf/inst_conf_new.json') as fd:
            self.__inst_conf = json.load(fd)
        with open('conf/detail_conf.json') as fd:
            self.__detail_conf = json.load(fd)
            self.max_fam, self.max_cpu, self.max_mem = -1, -1, -1
            for k, v in self.__detail_conf.items():
                self.max_fam = max(self.max_fam, v['family'])
                self.max_cpu = max(self.max_cpu, v['cpu'])
                self.max_mem = max(self.max_mem, v['memory'])
        with open('conf/global_max_vals.json') as fd:
            self.__global_max_vals = json.load(fd)

        self.runtime_settings = {}


    def runtime_set(self, *kv):
        assert len(kv) > 1, 'a value associated with this key must be indicated'
        runtime_key = '.'.join(kv[:-1])
        self.runtime_settings[runtime_key] = kv[-1]


    def get(self, *key):
        runtime_key = '.'.join(key)
        if runtime_key in self.runtime_settings:
            return self.runtime_settings[runtime_key]
        tmp = self.__conf[key[0]]
        if len(key) > 1:
            for i in range(len(key) - 1):
                tmp = tmp[key[i + 1]]
        return tmp


    def get_inst_id(self, inst):
        return self.__inst_conf[inst]

    
    def get_inst_detailed_conf(self, inst, format='dict'):
        v = self.__detail_conf[inst]
        if format == 'list':
            return [
                v['family'] / self.max_fam,
                v['cpu'] / self.max_cpu,
                v['memory'] / self.max_mem
            ]
        return {
            'family': v['family'] / self.max_fam,
            'cpu': v['cpu'] / self.max_cpu,
            'memory': v['memory'] / self.max_mem
        }


    def get_inst_hourly_price(self, inst):
        return self.__detail_conf[inst]['price']


    def get_scale_id(self, scale):
        scale_arr = ('tiny', 'small', 'large', 'huge')
        assert scale in scale_arr, 'invalid scale: %s' % scale
        return scale_arr.index(scale) / 4

    def get_global_max_val(self, idx, rnd='1'):
        return self.__global_max_vals[rnd][idx]
