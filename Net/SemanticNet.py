import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class NSMN(nn.Module):
	#内部方法
	def __init__(self, d0, d1, d2, d3, hidden_dim = 3, h_function_hidden_dim = 10, lstm_layers = 1, classify_num = 2, use_gpu = False, learning_rate = 0.01):
		super(NSMN, self).__init__()
		self.embedding_dim = d0
		#输出层类别数
		self.classify_num = classify_num
		# 网络中一些中间量的维度
		self.dim1 = d1
		self.dim2 = d2
		self.dim3 = d3
		#LSTM中中的隐藏层维度
		self.hidden_dim = hidden_dim
		#LSTM层数
		self.lstm_layers = lstm_layers
		# encoding层的LSTM
		self.encoding_lstm = nn.LSTM(d0, d1, batch_first = True, num_layers = lstm_layers)
		# matching层的LSTM
		self.matching_lstm = nn.LSTM(d2, d3, batch_first = True, num_layers = lstm_layers)
		#仿射函数
		self.f_function = nn.Linear(4*d1, d2)
		self.h_function_l1 = nn.Linear(4*d3, h_function_hidden_dim)
		self.h_function_l2 = nn.Linear(h_function_hidden_dim, classify_num)
		#配置
		self.gpu = use_gpu
		self.loss = nn.CrossEntropyLoss(reduce=True, size_average=True)
		self.opt = optim.SGD(self.parameters(), lr=learning_rate)
		pass
	
	def init_hidden(self, dim, batch_size):
		# (num_layers, minibatch_size, hidden_dim)
		if self.gpu:
			return (torch.zeros(self.lstm_layers, batch_size, dim).cuda(),torch.zeros(self.lstm_layers, batch_size, dim).cuda())
		else:
			return (torch.zeros(self.lstm_layers, batch_size, dim),torch.zeros(self.lstm_layers, batch_size, dim))
	
	def forward(self, U, V, batch_size):
		#状态初始化
		self.encoding_hidden = self.init_hidden(self.dim1, batch_size)
		self.matching_hidden = self.init_hidden(self.dim3, batch_size)
		# U V的形式
		# batch, seq_len, embedding_size
		# Encoding 层
		U_bar, _ = self.encoding_lstm(U, self.encoding_hidden)
		V_bar, _ = self.encoding_lstm(V, self.encoding_hidden) 
		U_bar = U_bar.permute(0,2,1) #batch, embedding_size, u_seq_len
		V_bar = V_bar.permute(0,2,1) #batch, embedding_size, v_seq_len
		# Alignment 层
		E = torch.matmul(U_bar.permute(0,2,1), V_bar) #batch u_seq_len v_seq_len
		U_wave = torch.matmul(V_bar, F.softmax(E.permute(0,2,1), dim = 1))
		V_wave = torch.matmul(U_bar, F.softmax(E, dim = 1))
		combined = torch.cat((U_bar, U_wave, U_bar - U_wave, U_bar * U_wave),1)
		combined = combined.permute(0,2,1)#batch, seq_len, embedding_size
		S = F.relu(self.f_function(combined))
		combined = torch.cat((V_bar, V_wave, V_bar - V_wave, V_bar * V_wave),1)
		combined = combined.permute(0,2,1)#batch, seq_len, embedding_size
		T = F.relu(self.f_function(combined))
		# Matching 层
		P, _ = self.matching_lstm(S, self.matching_hidden)
		Q, _ = self.matching_lstm(T, self.matching_hidden)
		# Output 层
		p, _ = torch.max(P, 1)
		q, _ = torch.max(Q, 1)
		combined = torch.cat((p, q, torch.abs(p-q), p*q),1)
		m = F.relu(self.h_function_l1(combined))
		m = F.relu(self.h_function_l2(m))
		m = F.softmax(m, dim=1)
		return m
	
	#可外部调用
	#训练一个batch [batch_size, seq_len, embed_size] input is numpy array
	def train_batch(self, samples_u, samples_v, sample_tags):
		batch_u = torch.from_numpy(samples_u).float()
		batch_v = torch.from_numpy(samples_v).float()
		batch_tags = torch.from_numpy(sample_tags).long()
		batch_size, _, _ =batch_u.size() 
		
		self.zero_grad()
		res = self.forward(batch_u,batch_v,batch_size)
		loss = self.loss(res, batch_tags)
		loss.backward()
		self.opt.step()
		return loss
	#infernce [seq_len, embed_size] input is numpy array
	def inference_once(self, sample_u, sample_v):
		inference_u = torch.from_numpy(sample_u[np.newaxis,:]).float()
		inference_v = torch.from_numpy(sample_v[np.newaxis,:]).float()

		res = self.forward(inference_u, inference_v, 1)
		res = res.detach().numpy()
		return res[0]

	def save_model(self, path):
		torch.save(self.state_dict(), path)
		return

	def load_model(self, path):
		self.load_state_dict(torch.load(path))
		return

	def generate_attack_sample(self, samples_u, samples_v, tag, epsilon = 0.3):
		inference_u = torch.from_numpy(samples_u).float()
		inference_v = torch.from_numpy(samples_v).float()
		inference_tag = torch.from_numpy(tag).long()
		batch_size, _, _ =inference_u.size() 

		inference_u.requires_grad = True
		inference_v.requires_grad = True

		res = self.forward(inference_u, inference_v, batch_size)
		loss = self.loss(res, inference_tag)
		self.zero_grad()
		loss.backward()

		u_grad = inference_u.grad.data
		v_grad = inference_v.grad.data
		
		attack_u = inference_u + epsilon * u_grad.sign()
		attack_v = inference_v + epsilon * v_grad.sign()

		return attack_u.detach().numpy(), attack_v.detach().numpy()

if __name__ == '__main__':
	#for test
	tag1 = np.array([1,1,1,1,1,1,1,1,1,1])
	tag2 = np.array([0,0,0,0,0,0,0,0,0,0])
	model = NSMN(5,4,4,3,hidden_dim = 3, lstm_layers = 3, classify_num = 2)
	model.load_model("./Net/trainedModel/testModel.pyt")
	tag = np.concatenate((tag1,tag2), axis = 0)

	for epoch in range(0):
		u = np.random.rand(10,4,5)
		v = np.random.rand(10,4,5)

		new_u = np.concatenate((u,u), axis = 0)
		new_v = np.concatenate((u,v), axis = 0)
		
		loss = model.train_batch(u, u, tag1)
		if epoch%50 == 0:
			print(epoch,": || loss:", loss)

	
	'''test_u = np.random.rand(4,5)
	test_v = np.random.rand(4,5)
	res = model.inference_once(test_u, test_v)
	print(res)
	res = model.inference_once(test_u, test_u)
	print(res)'''

	u = np.random.rand(10,4,5)
	v = np.random.rand(10,4,5)
	attack_u, attack_v = model.generate_attack_sample(u, v, tag1, epsilon = 0.8)
	print(u)
	print(attack_u)
	#model.save_model("./Net/trainedModel/testModel.pyt")

	'''attack_u, attack_v = model.generate_attack_sample(test_u, test_v, np.array([0]))
	res = model.inference_once(attack_u, attack_u)
	print(res, "----")
	for i in range(1):
		attack_u, attack_v = model.generate_attack_sample(attack_u, attack_v, np.array([0]), epsilon = 0.8)
		res = model.inference_once(attack_u, attack_u)
		if i % 20 == 0:
			print(res)'''
	#grad_u = torch.rand(3,4,5)
	#grad_v = torch.rand(3,4,5)
	#res = model.forward(grad_u,grad_v,3)
	#print(res)
	#grad_u.requires_grad = True
	#grad_v.requires_grad = True
	#res = model.forward(grad_u,grad_v,3)
	#print(res)
	pass