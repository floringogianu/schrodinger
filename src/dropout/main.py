import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from model import FCNet, ConvNet
import math

seed=1
num_epochs=3
lr=0.001
momentum=0.9
batch_size=64
log_interval=200
dropout=0.01 #0.1
tau=0.1 #0.2

torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 4, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
	datasets.MNIST('./data', train=True, download=True,
				   transform=transforms.Compose([
					   transforms.ToTensor(),
					   transforms.Normalize((0.1307,), (0.3081,))
				   ])),
	batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
	datasets.MNIST('./data', train=False, transform=transforms.Compose([
					   transforms.ToTensor(),
					   transforms.Normalize((0.1307,), (0.3081,))
				   ])),
	batch_size=1000, shuffle=True, **kwargs)

def train(model, criterion, optimizer):

	#train the net
	model.train()
	for epoch in range(num_epochs):
		for batch_idx, (data, target) in enumerate(train_loader):
			data, target = data.cuda(), target.cuda()
			data, target = Variable(data), Variable(target)
			optimizer.zero_grad()
			output = model(data)
			loss = criterion(F.log_softmax(output), target)
			loss.backward()
			optimizer.step()
			if batch_idx % log_interval == 0:
				print('Epoch: {} ({:.0f}%) \tLoss: {:.6f}'
						.format(epoch, 100. * batch_idx / len(train_loader), loss.data.item()))

	return model

def test(model):

	#test the net
	model.eval()
	test_loss = 0
	correct = 0
	for data, target in test_loader:
		data, target = data.cuda(), target.cuda()
		data, target = Variable(data, volatile=True), Variable(target)
		output = model(data)
		test_loss += F.nll_loss(F.log_softmax(output), target, size_average=False).data.item()  # sum up batch loss
		pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
		correct += pred.eq(target.data.view_as(pred)).cpu().sum()

	test_loss /= len(test_loader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
		test_loss, correct, len(test_loader.dataset),
		100. * correct / len(test_loader.dataset)))

def predict(model, x):

	model.train()
	T = 1000
	output_list = []
	for i in range(T):
		output_list.append(torch.unsqueeze(F.softmax(model(x)), 0))
	output_mean = torch.cat(output_list, 0).mean(0)
	output_variance = torch.cat(output_list, 0).var(0).mean().data.item()
	confidence = output_mean.data.cpu().numpy().max()
	label = output_mean.data.cpu().numpy().argmax()

	return output_variance, confidence, label

def main():
	
	# build and train model
	model=FCNet() #ConvNet()
	model.cuda()
	criterion = torch.nn.NLLLoss() #MSELoss()
	#reg = (1 - dropout) / (2. * len(train_loader) * tau)
	optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum) #weight_decay=reg
	train(model, criterion, optimizer)

	test(model)

	# rotation test
	rotation_list = range(0, 180, 10)
	for data, _ in test_loader:
		data = data.cuda()
		data = Variable(data, volatile=True)
		for x in data:
			x.unsqueeze_(0)
			for r in rotation_list:
				rotation_matrix = Variable(torch.Tensor([[[math.cos(r/360.0*2*math.pi), -math.sin(r/360.0*2*math.pi), 0],
														[math.sin(r/360.0*2*math.pi), math.cos(r/360.0*2*math.pi), 0]]]).cuda(),
										volatile=True)
				grid = F.affine_grid(rotation_matrix, x.size())
				x_rotate = F.grid_sample(x, grid)
				output_variance, confidence, label = predict(model, x_rotate)
				print ('rotation degree', str(r).ljust(3), 'Uncertainty : {:.4f} Label : {} Softmax : {:.2f}'.format(output_variance, label, confidence))

	'''
	plt.figure()
	for i in range(len(rotation_list)):
		ax = plt.subplot(2, len(rotation_list)/2, i+1)
		plt.text(0.5, -0.5, "{0:.3f}".format(unct_list[i]),
					size=12, ha="center", transform=ax.transAxes)
		plt.axis('off')
		plt.gca().set_title(str(rotation_list[i])+u'\xb0')
		plt.imshow(image_list[i][0, 0, :, :].data.cpu().numpy())
	plt.show()
	print ()
	'''

if __name__ == '__main__':
	main()