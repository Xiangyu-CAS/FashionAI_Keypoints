from __future__ import print_function
import os, sys
sys.path.append('../../')
import dataset_loader
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import util
import cv2
import argparse
import models.hourglass
import torchvision.transforms as transforms
import time

def parse():
    parser = argparse.ArgumentParser()
    return parser.parse_args()

def construct_model(args):
	model = models.hourglass.hg(num_stacks=2, num_blocks=1, num_classes=5)
	model.cuda()
	return model

def train_net(model, args):

	ann_path = '../FashionAI/data/train/Annotations/train.csv'
	img_dir = '../FashionAI/data/train/'

	stride = 8
	cudnn.benchmark = True
	config = util.Config('./config.yml')

	train_loader = torch.utils.data.DataLoader(
		dataset_loader.dataset_loader(img_dir, ann_path, stride,
		                              transforms.ToTensor()),
		batch_size=config.batch_size, shuffle=True,
		num_workers=config.workers, pin_memory=True)

	criterion = nn.MSELoss().cuda()

	optimizer = torch.optim.SGD(model.parameters(), config.base_lr, momentum=config.momentum,
	                            weight_decay=config.weight_decay)
	model.train()
	iters = 0
	batch_time = util.AverageMeter()
	data_time = util.AverageMeter()
	losses = util.AverageMeter()
	losses_list = [util.AverageMeter() for i in range(12)]
	end = time.time()

	heat_weight = 48 * 48 * 25 / 2.0  # for convenient to compare with origin code
	# heat_weight = 1

	while iters < config.max_iter:
		for i, (input, heatmap) in enumerate(train_loader):
			learning_rate = util.adjust_learning_rate(optimizer, iters, config.base_lr, policy=config.lr_policy,\
								policy_parameter=config.policy_parameter)
			data_time.update(time.time() - end)

			input = input.cuda(async=True)
			heatmap = heatmap.cuda(async=True)
			input_var = torch.autograd.Variable(input)
			heatmap_var = torch.autograd.Variable(heatmap)

			output = model(input_var)
			loss = criterion(output[0], heatmap_var)* heat_weight
			for j in range(1, len(output)):
				loss += criterion(output[j], heatmap_var) * heat_weight

			losses.update(loss.data[0], input.size(0))
			loss_list = loss
			for cnt, l in enumerate(loss_list):
				losses_list[cnt].update(l.data[0], input.size(0))

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			batch_time.update(time.time() - end)
			end = time.time()


			iters += 1
			if iters % config.display == 0:
				print('Train Iteration: {0}\t'
				      'Time {batch_time.sum:.3f}s / {1}iters, ({batch_time.avg:.3f})\t'
				      'Data load {data_time.sum:.3f}s / {1}iters, ({data_time.avg:3f})\n'
				      'Learning rate = {2}\n'
				      'Loss = {loss.val:.8f} (ave = {loss.avg:.8f})\n'.format(
					iters, config.display, learning_rate, batch_time=batch_time,
					data_time=data_time, loss=losses))
				for cnt in range(0, 1):
					print('Loss{0}_1 = {loss1.val:.8f} (ave = {loss1.avg:.8f})'.format(cnt + 1,loss1=losses_list[cnt]))
				print(time.strftime(
					'%Y-%m-%d %H:%M:%S -----------------------------------------------------------------------------------------------------------------\n',
					time.localtime()))

				batch_time.reset()
				data_time.reset()
				losses.reset()
				for cnt in range(12):
					losses_list[cnt].reset()

			if iters % 5000 == 0:
				torch.save({
					'iter': iters,
					'state_dict': model.state_dict(),
				},  str(iters) + '.pth.tar')

			if iters == config.max_iter:
				break
	return

if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'
	args = parse()
	model = construct_model(args)
	train_net(model, args)