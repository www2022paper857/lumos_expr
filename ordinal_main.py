import os
import sys
import json
import dill
import random
import pickle
import numpy as np

from utils import *
from conf import LumosConf
from collections import defaultdict
from data_loader_ordinal import DataLoaderOrdinal
from third_party.keras_lstm_vae.lstm_vae import create_lstm_vae


if __name__ == "__main__":
    conf = LumosConf()
    dump_pth = conf.get('dataset', 'dump_pth_ordinal')
    dataloader = DataLoaderOrdinal(dump_pth=dump_pth)
    dataloader.load_data()