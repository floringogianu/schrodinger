import torch.nn as nn
import torch.nn.functional as F

dropout=0.1

class FCNet(nn.Module):

	def __init__(self):
		super(FCNet, self).__init__()
		self.fc1 = nn.Linear(28*28, 128)
		self.fc2 = nn.Linear(128, 128)
		self.fc3 = nn.Linear(128, 10)

	def forward(self, x):
		x = x.view(-1, 28*28)
		x = F.dropout(F.relu(self.fc1(x)), p=dropout, training=self.training)
		x = F.dropout(F.relu(self.fc2(x)), p=dropout, training=self.training)
		x = self.fc3(x)
		return x

class ConvNet(nn.Module):

	def __init__(self):
		super(ConvNet, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16*4*4, 50)
		#self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(50, 10)

	def forward(self, x):
		x = F.relu(F.max_pool2d(F.dropout(self.conv1(x), p=dropout, training=self.training), 2))
		x = F.relu(F.max_pool2d(F.dropout(self.conv2(x), p=dropout, training=self.training), 2))
		x = x.view(-1, 16*4*4)
		x = F.dropout(F.relu(self.fc1(x)), training=self.training)
		#x = F.dropout(F.relu(self.fc2(x)), training=self.training)
		x = self.fc3(x)
		return x
