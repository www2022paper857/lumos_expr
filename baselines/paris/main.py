# -*- coding: utf-8 -*-

import numpy as np  
import json, os
from sklearn.ensemble import RandomForestRegressor  

# 'small', 'large' or 'all'
train_scale = 'all'
predict_scale = 'large'
normalize_rate = 30
# choose 2 instance types as reference, use runtime on them to predict runtime on other VMs
reference_instances = ['hfr6.2xlarge', 's6.2xlarge.2']
n_estimators = 10

def get_features():
	"""
	Get performance metrics of all workloads on all VMs
	Change Path to lumos-dataset
	"""
	features = {}
	confs = {}
	with open('detail_conf.json') as f:
		confs = json.load(f)
	Path = '/Users/macpro/Desktop/lumos-dataset/'
	vendors = ['alibaba/1 second/', 'huawei/1 second/', 'tencent/1 second/', 'aws/']
	vendors2 = ['alibaba/5 second/', 'huawei/5 second/', 'tencent/5 second/'] 
	for vendor in vendors:
		for instance in os.listdir(Path + vendor):
			if instance[0] == '.':
				continue
			path = Path + vendor + instance
			for bench in os.listdir(path):
				if bench[-2:] == '_2' or bench[-2:] == '_3':
					continue
				if bench[0] == '.' or bench[-1] in ['4', '5', '6']:
					continue
				with open(path + '/' + bench + '/report.json') as f:
					result = json.load(f)
					if not bench in features:
						features[bench] = {instance: [float(result['elapsed_time'])]}
					else:
						features[bench][instance] = [float(result['elapsed_time'])]
				with open(path + '/' + bench + '/sar.csv') as f:
					lines = f.readlines()[1:-1]
					l = len(lines)
					tmp = lines[int(l/4)].strip().split(',')[1:]
					tmp.remove('lo')
					tmp.extend(lines[int(2*l/4)].strip().split(',')[1:])
					tmp.remove('lo')
					tmp.extend(lines[int(3*l/4)].strip().split(',')[1:])
					tmp.remove('lo')
					features[bench][instance].extend(tmp)
	for vendor in vendors2:
		for instance in os.listdir(Path + vendor):
			if instance[0] == '.':
				continue
			path = Path + vendor + instance
			for bench in os.listdir(path):
				if bench[-2:] == '_2' or bench[-2:] == '_3':
					continue
				if bench[0] == '.' or bench[-1] in ['4', '5', '6'] or (('hadoop' in bench or 'spark' in bench) and ('tiny' in bench or 'small' in bench)):
					continue
				with open(path + '/' + bench + '/report.json') as f:
					result = json.load(f)
					if not bench in features:
						features[bench] = {instance: [float(result['elapsed_time'])]}
					else:
						features[bench][instance] = [float(result['elapsed_time'])]
				with open(path + '/' + bench + '/sar.csv') as f:
					lines = f.readlines()[1:-1]
					l = len(lines)
					tmp = lines[int(l/4)].strip().split(',')[1:]
					tmp.remove('lo')
					tmp.extend(lines[int(2*l/4)].strip().split(',')[1:])
					tmp.remove('lo')
					tmp.extend(lines[int(3*l/4)].strip().split(',')[1:])
					tmp.remove('lo')
					features[bench][instance].extend(tmp)
	return features, confs

def handle_features(features, confs):
	"""
	get train data X, y from original features
	return costs: cost of instance of i-th train data 
	       benches: benchname of i-th train data 
	"""
	X = []
	y = []
	costs = []
	benches = []
	for bench, datas in features.items():
		for instance, data in datas.items():
			# target instance information
			tmp = list(confs[instance].values())[0:3]
			# performance metrics on two reference VMs
			tmp.extend(datas[reference_instances[0]])
			tmp.extend(datas[reference_instances[1]])
			X.append(tmp)
			# normalized time
			y.append(data[0]*normalize_rate/datas[reference_instances[0]][0])
			costs.append(list(confs[instance].values())[3])
			benches.append(bench)
	return X, y, costs, benches

def get_accuracy(train_X, train_y, test_X, test_y, benches, costs, same=False):
	"""
	calculate jct accuracy and cost accuracy
	top: top n
	same: whether the train_data and predict_data are the same, 
		if true, extract the using one predict_data from train_data
		else, append two datas on reference VMs from predict_data to train_data 
	"""
	total = 0
	jct_correct1 = 0
	jct_correct2 = 0
	cost_correct1 = 0
	cost_correct2 = 0
	jct_absolute = []
	jct_relative = []
	cost_absolute = []
	cost_relative = []
	rf = RandomForestRegressor(n_estimators=n_estimators, random_state=0) 
	all_bench = list(set([benches[t[0]] for t in test_y]))
	for bench in all_bench:
		rf.fit([t[1] for t in train_X if '_'.join(benches[t[0]].split('_')[:2]) != '_'.join(bench.split('_')[:2])], [t[1] for t in train_y if '_'.join(benches[t[0]].split('_')[:2]) != '_'.join(bench.split('_')[:2])])
		predict_X = [t[1] for t in test_X if benches[t[0]] == bench]
		real_y = [t[1] for t in test_y if benches[t[0]] == bench]
		tmpcosts = [costs[t[0]] for t in test_y if benches[t[0]] == bench]
		predict_y = rf.predict(predict_X)
		
		chooseninstance = np.argmin(np.array(predict_y))
		result = real_y[chooseninstance]

		real_min = np.min(real_y)
		jct_absolute.append(round(result - real_min, 3))
		jct_relative.append(round((result - real_min)/real_min, 3))

		count = 0
		flag1 = True
		flag2 = True
		for time in real_y:
			if time < result:
				count += 1
			# threshold
			if count > 5:
				flag2 = False
				break
			if count > 3:
				flag1 = False
		if flag1:
			jct_correct1 += 1
		if flag2:
			jct_correct2 += 1

		chooseninstance = np.argmin(np.array(predict_y)*np.array(tmpcosts))
		result = real_y[chooseninstance] * tmpcosts[chooseninstance]

		cost_min = np.min(np.array(real_y)*np.array(tmpcosts))
		cost_absolute.append(round(result - cost_min, 3))
		cost_relative.append(round((result - cost_min)/cost_min, 3))

		count = 0
		flag1 = True
		flag2 = True
		for j, time in enumerate(real_y):
			if time * tmpcosts[j] < result:
				count += 1
			# threshold
			if count > 5:
				flag2 = False
				break
			if count > 3:
				flag1 = False
		if flag1:
			cost_correct1 += 1
		if flag2:
			cost_correct2 += 1

		total += 1
	
	ret = [round(jct_correct1/total, 3), round(cost_correct1/total, 3), round(jct_correct2/total, 3), round(cost_correct2/total, 3), jct_absolute, jct_relative, cost_absolute, cost_relative]
	return all_bench, ret

if __name__ == '__main__':
	features, confs = get_features()
	X, y, costs, benches = handle_features(features, confs)
	X = [(i,t) for i, t in enumerate(X) if 'hive' in benches[i]]
	y = [(i,t) for i, t in enumerate(y) if 'hive' in benches[i]]
	# X2 = [(i,t) for i, t in enumerate(X) if 'hive' in benches[i]]
	# y2 = [(i,t) for i, t in enumerate(y) if 'hive' in benches[i]]
	# X = [(i,t) for i, t in enumerate(X) if 'hadoop' in benches[i] or 'spark' in benches[i]]
	# y = [(i,t) for i, t in enumerate(y) if 'hadoop' in benches[i] or 'spark' in benches[i]]
	flag = True
	train_X = []; train_y = []
	test_X = []; test_y = []
	if train_scale == 'all':
		train_X = [(t[0],t[1]) for t in X if not 'tiny' in benches[t[0]]]
		train_y = [(t[0],t[1]) for t in y if not 'tiny' in benches[t[0]]]
	else:
		train_X = [(t[0],t[1]) for t in X if train_scale in benches[t[0]]]
		train_y = [(t[0],t[1]) for t in y if train_scale in benches[t[0]]]
	test_X = [(t[0],t[1]) for t in X if predict_scale in benches[t[0]]]
	test_y = [(t[0],t[1]) for t in y if predict_scale in benches[t[0]]]
	if (train_scale != predict_scale and train_scale != 'all') or flag:
		bench_names, ret = get_accuracy(train_X, train_y, test_X, test_y, benches, costs)
	else:
		bench_names, ret = get_accuracy(train_X, train_y, test_X, test_y, benches, costs, True)
	out = {}
	out['top3'] = {'jct': ret[0], 'cost': ret[1]}
	out['top5'] = {'jct': ret[2], 'cost': ret[3]}
	for i, t in enumerate(bench_names):
		t = '_'.join(t.split('_')[:2])
		out[t] = {'jct_absolute': ret[4][i], 'jct_relative': ret[5][i], 'cost_absolute': ret[6][i], 'cost_relative': ret[7][i]}
	with open('tmp.json', 'w') as f:
		json.dump(out, f)
