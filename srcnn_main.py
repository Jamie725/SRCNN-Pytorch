from PIL import Image as image
from PIL import ImageFilter as IF
from image_utils import *
from srcnn_module import SRCNN
from math import log10
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch 
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import numpy as np
#=====Parameters======#
upscale_factor = 4
epoch = 20
learn_rate = 0.0001
use_gpu = 1
trainLR = []
trainHR = []
train_path = 'dataset/BSDS300/images/train'
test_path = 'dataset/BSDS300/images/test'
baseName = 'output/out_'
#=====================#

transform_data = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

dl_dataset()

def transform(img):
	crop = transforms.CenterCrop((int(img.size[1]/upscale_factor)*upscale_factor,int(img.size[0]/upscale_factor)*upscale_factor))
	img = crop(img)
	out = img.filter(IF.GaussianBlur(1.3))#.convert('YCbCr')
	out = out.resize((int(out.size[0]/upscale_factor), int(out.size[1]/upscale_factor)))	
	out = out.resize((int(out.size[0]*upscale_factor), int(out.size[1]*upscale_factor)))
	return transform_data(out), transform_data(img)

trainSet = datasets.ImageFolder(train_path, transform=transform, target_transform=None)
testSet = datasets.ImageFolder(test_path, transform=transform, target_transform=None)

srcnn = SRCNN()
loss_func = nn.MSELoss()

if use_gpu :
	srcnn.cuda()
	loss_func = loss_func.cuda()

optimizer = opt.Adam(srcnn.parameters(), lr = learn_rate)

def train(epoch, trainSet):
	epoch_loss = 0
	for itr, data in enumerate(trainSet):
		imgs, label = data
		imgLR, imgHR = imgs
		imgLR.unsqueeze_(0)
		imgHR.unsqueeze_(0)
		
		if use_gpu :
			imgLR = imgLR.cuda()
			imgHR = imgLR.cuda()

		optimizer.zero_grad()
		out_model = srcnn(imgLR)	
		loss = loss_func(out_model, imgHR)
		epoch_loss += loss.item()
		loss.backward()
		optimizer.step()
		#print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, itr, len(trainSet), loss.item()))
	print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(trainSet)))

def test(epoch, testSet, saveImgFlag):
	sum_psnr = 0
	for itr, data in enumerate(testSet):
		imgs, label = data
		imgLR, imgHR = imgs
		imgLR.unsqueeze_(0)
		imgHR.unsqueeze_(0)
		
		if use_gpu :
			imgLR = imgLR.cuda()
			imgHR = imgLR.cuda()

		sr_result = srcnn(imgLR)

		if use_gpu:
			outImg = sr_result.data.cpu().squeeze(0)
		else:
			outImg = sr_result.data.squeeze(0)
			
		if saveImgFlag:
			outFileName = baseName + 'epoch_' + str(epoch) + '_' + str(itr) + '.jpg'
			saveImg(outImg, outFileName)
		
		MSE = loss_func(sr_result, imgHR)
		psnr = 10*log10(1/MSE.item())
		sum_psnr += psnr
	print("**Average PSNR: {} dB".format(sum_psnr/len(testSet)))
	#return outImg

outImg = []

for epoch in range(1, epoch+1):
	train(epoch, trainSet)
	test(epoch, testSet, 0)
	#outFileName = baseName + 'epoch_' + str(epoch) + '.jpg'
	#saveImg(outImg, outFileName)

test(epoch, testSet, 1)
