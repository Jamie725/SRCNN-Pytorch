from srcnn_module import *
import torchvision.transforms as transforms
import torchvision.datasets as datasets

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

def test(testSet):
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
		MSE = loss_func(sr_result, imgHR)
		psnr = 10*log10(1/MSE.item())
		sum_psnr += psnr
	print("**Average PSNR: {} dB".format(sum_psnr/len(testSet)))
