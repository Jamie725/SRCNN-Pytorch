from PIL import Image as image
import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
	def __init__(self):
		super(SRCNN, self).__init__()
		self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3);
		self.conv2 = nn.Conv2d(64, 32, kernel_size=9, padding=4);
		self.conv3 = nn.Conv2d(32, 32, kernel_size=9, padding=4);
		self.conv4 = nn.Conv2d(32, 3, kernel_size=7, padding=3);
		#self.relu  = nn.ReLU();

	def forward(self, img):
		out = F.relu(self.conv1(img))
		out = F.relu(self.conv2(out))
		out = F.relu(self.conv3(out))
		out = self.conv4(out)
		return out