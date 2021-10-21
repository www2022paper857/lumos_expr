# -*- coding: utf-8 -*-

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os, json

def getfeatures():
	features = {}
	Path = '/Users/macpro/Desktop/lumos-dataset/'
	vendors = ['alibaba/1 second/', 'huawei/1 second/', 'tencent/1 second/', 'aws/']
	vendors2 = ['alibaba/5 second/', 'huawei/5 second/', 'tencent/5 second/'] 
	for vendor in vendors:
		for instance in os.listdir(Path + vendor):
			if instance[0] == '.':
				continue
			path = Path + vendor + instance
			for bench in os.listdir(path):
				if bench[0] == '.' or bench[-1] in ['2', '3', '4', '5', '6']:
					continue
				with open(path + '/' + bench + '/report.json') as f:
					result = json.load(f)
					if not bench in features:
						features[bench] = {instance: [float(result['elapsed_time'])]}
					else:
						features[bench][instance] = [float(result['elapsed_time'])]
				with open(path + '/' + bench + '/sar.csv') as f:
					lines = f.readlines()[1:]
					for i in range(len(lines)):
						lines[i] = lines[i].strip().split(',')[1:]
						lines[i].remove('lo')
						for j in range(len(lines[i])):
							lines[i][j] = float(lines[i][j])
					features[bench][instance].append(lines)
	for vendor in vendors2:
		for instance in os.listdir(Path + vendor):
			if instance[0] == '.':
				continue
			path = Path + vendor + instance
			for bench in os.listdir(path):
				if bench[0] == '.' or bench[-1] in ['2', '3', '4', '5', '6'] or (('hadoop' in bench or 'spark' in bench) and ('tiny' in bench or 'small' in bench)):
					continue
				with open(path + '/' + bench + '/report.json') as f:
					result = json.load(f)
					if not bench in features:
						features[bench] = {instance: [float(result['elapsed_time'])]}
					else:
						features[bench][instance] = [float(result['elapsed_time'])]
				with open(path + '/' + bench + '/sar.csv') as f:
					lines = f.readlines()[1:]
					for i in range(len(lines)):
						lines[i] = lines[i].strip().split(',')[1:]
						lines[i].remove('lo')
						for j in range(len(lines[i])):
							lines[i][j] = float(lines[i][j])
					features[bench][instance].append(lines)
	return features

# torch.manual_seed(1)    # reproducible

# Hyper Parameters
INPUT_SIZE = 61     # rnn input size
LR = 0.02           # learning rate
normalize_rate = 30
reference_instance = 'hfr6.2xlarge'

class RNN(nn.Module):
	def __init__(self):
		super(RNN, self).__init__()

		self.rnn = nn.LSTM(
			input_size=INPUT_SIZE,
			hidden_size=32,     # rnn hidden unit
			num_layers=1,       # number of rnn layer
			batch_first=True,   # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
		)
		self.hidden1 = nn.Linear(32, 1)
		self.hidden2 = nn.Linear(6, 1)
		self.out = nn.Linear(4, 1)

	def forward(self, x, x2, x3, x4, x5):
		# x shape (batch, time_step, input_size)
		# r_out shape (batch, time_step, output_size)
		# h_n shape (n_layers, batch, hidden_size)
		# h_c shape (n_layers, batch, hidden_size)
		r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
		# choose r_out at the last time step
		tmp1 = self.hidden1(r_out[:, -1, :])
		tmp2 = self.hidden2(torch.cat([x2, x3], dim=1))
		out = self.out(torch.cat([tmp1, tmp2, x4, x5], dim=1))
		return out

def train(features, confs):
	rnn = RNN()

	optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)   # optimize all cnn parameters
	loss_func = nn.MSELoss()

	i = 0
	for bench, benchdata in features.items():
		if not 'small' in bench or not 'hive' in bench:
			continue 
		benchlarge = bench.replace('small', 'large')
		benchhuge = bench.replace('small', 'huge')
		i += 1
		j = 0
		for instance, data in benchdata.items():
			x = torch.from_numpy(np.array(data[1], dtype=np.float32)).view(1, -1, INPUT_SIZE)    # shape (batch, time_step, input_size)
			x2 = torch.from_numpy(np.array(list(confs[instance].values())[:3], dtype=np.float32)).view(1, 3)
			x5 = torch.tensor([[data[0]*normalize_rate/benchdata[reference_instance][0]]])
			print(i, j)
			j += 1
			for instance2, data2 in benchdata.items():
				y = torch.tensor([[data2[0]*normalize_rate/benchdata[reference_instance][0]]])
				x3 = torch.from_numpy(np.array(list(confs[instance2].values())[:3], dtype=np.float32)).view(1, 3)
				x4 = torch.tensor([[1.0]])

				prediction = rnn(x, x2, x3, x4, x5)   # rnn output
				loss = loss_func(prediction, y)         # calculate loss
				optimizer.zero_grad()                   # clear gradients for this training step
				loss.backward()                         # backpropagation, compute gradients
				optimizer.step()                        # apply gradients

			if benchlarge in features:
				for instance2, data2 in features[benchlarge].items():
					y = torch.tensor([[data2[0]*normalize_rate/features[benchlarge][reference_instance][0]]])
					x3 = torch.from_numpy(np.array(list(confs[instance2].values())[:3], dtype=np.float32)).view(1, 3)
					x4 = torch.tensor([[2.0]])
					
					prediction = rnn(x, x2, x3, x4, x5)   # rnn output
					loss = loss_func(prediction, y)         # calculate loss
					optimizer.zero_grad()                   # clear gradients for this training step
					loss.backward()                         # backpropagation, compute gradients
					optimizer.step()                        # apply gradients
			
			if benchhuge in features:
				for instance2, data2 in features[benchhuge].items():
					y = torch.tensor([[data2[0]*normalize_rate/features[benchhuge][reference_instance][0]]])
					x3 = torch.from_numpy(np.array(list(confs[instance2].values())[:3], dtype=np.float32)).view(1, 3)
					x4 = torch.tensor([[3.0]])
					
					prediction = rnn(x, x2, x3, x4, x5)   # rnn output
					loss = loss_func(prediction, y)         # calculate loss
					optimizer.zero_grad()                   # clear gradients for this training step
					loss.backward()                         # backpropagation, compute gradients
					optimizer.step()                        # apply gradients

	torch.save(rnn, 'lstm_bigdatabench.pkl') 

def get_accuracy(features, confs, rnn):
	total = 0
	jct_correct1 = 0
	jct_correct2 = 0
	cost_correct1 = 0
	cost_correct2 = 0
	jct_absolute = []
	jct_relative = []
	cost_absolute = []
	cost_relative = []
	bench_names = []
	for bench, benchdata in features.items():
		if not 'small' in bench or not 'hive' in bench:
			continue 
		instance = reference_instance
		benchsmall = bench.replace('small', 'small')
		if not benchsmall in features:
			continue
		data = features[benchsmall][instance]
		x = torch.from_numpy(np.array(data[1], dtype=np.float32)).view(1, -1, INPUT_SIZE)    # shape (batch, time_step, input_size)
		x2 = torch.from_numpy(np.array(list(confs[instance].values())[:3], dtype=np.float32)).view(1, 3)
		x5 = torch.tensor([[data[0]*normalize_rate/data[0]]])

		bench_names.append('_'.join(bench.split('_')[:2]))
		predictions = []
		real = []
		tmpcosts = []
		for instance2, data2 in benchdata.items():
			real.append(data2[0]*normalize_rate/data[0])
			tmpcosts.append(confs[instance2]['price'])
			x3 = torch.from_numpy(np.array(list(confs[instance2].values())[:3], dtype=np.float32)).view(1, 3)
			x4 = torch.tensor([[2.0]])
			prediction = rnn(x, x2, x3, x4, x5)   # rnn output
			predictions.append(float(prediction.data[0][0]))
		chooseninstance = np.argmin(np.array(predictions))
		result = real[chooseninstance]

		real_min = np.min(real)
		jct_absolute.append(round(result - real_min, 3))
		jct_relative.append(round((result - real_min)/real_min, 3))

		count = 0
		flag1 = True
		flag2 = True
		for time in real:
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

		chooseninstance = np.argmin(np.array(predictions)*np.array(tmpcosts))
		result = real[chooseninstance] * tmpcosts[chooseninstance]

		cost_min = np.min(np.array(real)*np.array(tmpcosts))
		cost_absolute.append(round(result - cost_min, 3))
		cost_relative.append(round((result - cost_min)/cost_min, 3))

		count = 0
		flag1 = True
		flag2 = True
		for j, time in enumerate(real):
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
	return bench_names, ret

if __name__ == '__main__':
	features = getfeatures()
	with open('detail_conf.json') as f:
		confs = json.load(f)
	# train(features, confs)
	rnn = torch.load('lstm_bigdatabench.pkl')
	bench_names, ret = get_accuracy(features, confs, rnn)
	out = {}
	out['top3'] = {'jct': ret[0], 'cost': ret[1]}
	out['top5'] = {'jct': ret[2], 'cost': ret[3]}
	for i, t in enumerate(bench_names):
		out[t] = {'jct_absolute': ret[4][i], 'jct_relative': ret[5][i], 'cost_absolute': ret[6][i], 'cost_relative': ret[7][i]}
	with open('tmp.json', 'w') as f:
		json.dump(out, f)