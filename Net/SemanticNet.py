import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NSMN(nn.Module):
	def __init__(self, d0, d1, d2, d3, hidden_dim = 3, lstm_layers = 1, classify_num = 2, use_gpu = False):
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
		self.h_function_l1 = nn.Linear(4*d3, 3)
		self.h_function_l2 = nn.Linear(3, classify_num)
		#配置
		self.gpu = use_gpu
		self.loss = nn.CrossEntropyLoss(reduce=True, size_average=True)
		self.opt = optim.SGD(self.parameters(), lr=0.1)
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

	def train_batch(self, batch_u, batch_v, batch_tags, learning_rate = 0.1):
		batch_size, _, _ =batch_u.size() 
		
		self.zero_grad()
		res = self.forward(batch_u,batch_v,batch_size)
		loss = self.loss(res, batch_tags)
		loss.backward()
		self.opt.step()
		return

if __name__ == '__main__':
	tag1 = torch.tensor([1,1,1,1,1,1,1,1,1,1]).cuda()
	tag2 = torch.tensor([0,0,0,0,0,0,0,0,0,0]).cuda()
	model = NSMN(5,4,4,3,hidden_dim = 3, lstm_layers = 2, classify_num = 2, use_gpu = True).cuda()

	u = torch.rand(10,4,5).cuda()
	v = torch.rand(10,4,5).cuda()

	new_u = torch.cat((u,u),0)
	new_v = torch.cat((u,v),0)
	tag = torch.cat((tag1,tag2),0)
	for epoch in range(10):	
		model.train_batch(new_u, new_v, tag)

	res = model(u,v,10)
	print(res)
	res = model(u,u,10)
	print(res)

	pass