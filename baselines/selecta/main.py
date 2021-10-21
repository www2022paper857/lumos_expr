# -*- coding: utf-8 -*-

import os, json
import collections
import numpy as np
import pandas as pd
from surprise import SVD
from surprise import Dataset, Reader
from surprise.model_selection import cross_validate

def get_times():
	"""
	Get runtimes of all workloads on all VMs
	Return instances and costs as a standard order, SVD use orderID to identify instances and workloads
	Change Path to lumos-dataset
	"""
	result_times = {}
	instances = []
	costs = []
	confs = {}
	with open('conf/detail_conf.json') as f:
		confs = json.load(f)
	Path = '/Users/macpro/Desktop/lumos-dataset/'
	vendors = ['alibaba/1 second/', 'huawei/1 second/', 'tencent/1 second/', 'aws/']
	vendors2 = ['alibaba/5 second/', 'huawei/5 second/', 'tencent/5 second/'] 
	for vendor in vendors:
		for instance in os.listdir(Path + vendor):
			if instance[0] == '.':
				continue
			path = Path + vendor + instance
			v = vendor.split('/')[0]+'/'
			instances.append(v + instance)
			costs.append(list(confs[instance].values())[3])
			for bench in os.listdir(path):
				if bench[-2:] == '_2' or bench[-2:] == '_3':
					continue
				if bench[0] == '.' or bench[-1] in ['4', '5', '6']:
					continue
				with open(path + '/' + bench + '/report.json') as f:
					result = json.load(f)
					if not bench in result_times:
						result_times[bench] = {v+instance: float(result['elapsed_time'])}
					else:
						result_times[bench][v+instance] = float(result['elapsed_time'])
	for vendor in vendors2:
		for instance in os.listdir(Path + vendor):
			if instance[0] == '.':
				continue
			path = Path + vendor + instance
			v = vendor.split('/')[0]+'/'
			for bench in os.listdir(path):
				if bench[-2:] == '_2' or bench[-2:] == '_3':
					continue
				if bench[0] == '.' or bench[-1] in ['4', '5', '6'] or (('hadoop' in bench or 'spark' in bench) and ('tiny' in bench or 'small' in bench)):
					continue
				with open(path + '/' + bench + '/report.json') as f:
					result = json.load(f)
					if not bench in result_times:
						result_times[bench] = {v+instance: float(result['elapsed_time'])}
					else:
						result_times[bench][v+instance] = float(result['elapsed_time'])
	return result_times, instances, costs

def normalize(p, benches, instances):
	"""
	normalize runtime according to reference VM alibaba/hfr6.2xlarge
	('user', 'item', 'rating') -> ('bench', 'instance', 'time')
	p: result['alibaba/hfr6.2xlarge'] / p as 1
	"""
	datas = []
	for bench, result in bench_times.items():
		baseline = result['alibaba/hfr6.2xlarge'] / p
		for instance, time in result.items():
			time = time / baseline
			datas.append([str(benches.index(bench)), str(instances.index(instance)), str(time)])
	return datas

def train_model(data):
	"""
	train data using SVD model
	"""
	ratings_dict = {'user': data[:, 0], 'item': data[:, 1], 'rating': data[:, 2]}
	df = pd.DataFrame(ratings_dict)

	reader = Reader(line_format='user item rating', sep=',', rating_scale=(1, 100))
	data = Dataset.load_from_df(df[['user', 'item', 'rating']], reader=reader)
	trainset = data.build_full_trainset()
	svd = SVD()
	svd.fit(trainset)
	return svd

def get_accuracy(train_data, predict_data, costs, top, same=False):
	"""
	calculate jct accuracy and cost accuracy
	top: top n
	same: whether the train_data and predict_data are the same, 
		if true, extract the using one predict_data from train_data
		else, append two datas on reference VMs from predict_data to train_data 
	"""
	# choose 2 instance types as reference, use runtime on them to predict runtime on other VMs
	reference_instance = [str(instances.index('alibaba/hfr6.2xlarge')), str(instances.index('tencent/s5.large8'))]
	total = 0
	jct_correct = 0
	cost_correct = 0
	benches = set([t[0] for t in predict_data])
	for i in benches:
		data = []
		if not same:
			data = train_data[:]
			data.extend([t for t in predict_data if t[0] == i and t[1] in reference_instance])
		else:
			data = [t for t in train_data if t[0] != i or t[1] in reference_instance]
		svd = train_model(np.array(data))

		real_times = [float(t[2]) for t in predict_data if t[0] == i]
		vms = [t[1] for t in predict_data if t[0] == i]
		predict_times = []
		tmpcosts = []
		for j in vms:
			pred = svd.predict(i, j)
			predict_times.append(pred.est)
			tmpcosts.append(costs[int(j)])
		
		chooseninstance = np.argmin(np.array(predict_times))
		result = real_times[chooseninstance]

		count = 0
		flag = True
		for time in real_times:
			if time < result:
				count += 1
			# threshold
			if count > top:
				flag = False
				break
		if flag:
			jct_correct += 1

		chooseninstance = np.argmin(np.array(predict_times)*np.array(tmpcosts))
		result = real_times[chooseninstance] * tmpcosts[chooseninstance]

		count = 0
		flag = True
		for j, time in enumerate(real_times):
			if time * tmpcosts[j] < result:
				count += 1
			# threshold
			if count > top:
				flag = False
				break
		if flag:
			cost_correct += 1

		total += 1
	return jct_correct / total, cost_correct / total

if __name__ == '__main__':
	bench_times, instances, costs = get_times()
	benches = [bench for bench in bench_times.keys()]
	datas = normalize(30, benches, instances)
	# 'small', 'large' or 'all'
	train_scale = 'large'
	predict_scale = 'large'
	top = 3
	train_data = []; predict_data = []
	if train_scale == 'all':
		train_data = [t for t in datas if 'small' in benches[int(t[0])] or 'large' in benches[int(t[0])]]
	else:
		train_data = [t for t in datas if train_scale in benches[int(t[0])]]
	if predict_scale == 'all':
		predict_data = [t for t in datas if 'small' in benches[int(t[0])] or 'large' in benches[int(t[0])]]
	else:
		predict_data = [t for t in datas if predict_scale in benches[int(t[0])]]
	if train_scale != predict_scale:
		# predicting workload not in training workloades
		jct_accuracy, cost_accuracy = get_accuracy(train_data, predict_data, costs, top)
	else:
		jct_accuracy, cost_accuracy = get_accuracy(train_data, predict_data, costs, top, True)
	print(jct_accuracy, cost_accuracy)

"""
Results
small -> large
top1: 0.57, 0.90; top3: 0.63, 0.93; top5: 0.83, 0.97
all -> all
top1: 0.49, 0.93; top3: 0.59, 0.98; top5: 0.77, 0.98
large -> large
top1: 0.60, 0.90; top3: 0.67, 0.93; top5: 0.87, 0.93
"""
