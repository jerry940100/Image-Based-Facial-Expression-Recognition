import torchvision.models as models
import torch
import torch.nn as nn
class MobileNet_Extractor(nn.Module):
	def __init__(self,pretrained=True):
		super().__init__()
		self.mobilenet=models.mobilenet_v3_large(pretrained)
		del self.mobilenet.classifier
		features = list(self.mobilenet.features)
		features.append(self.mobilenet.avgpool)
		self.layers = nn.ModuleList(features).eval()
	def forward(self,x):
		for ii,model in enumerate(self.layers):
			x = model(x)
		return x

class Encoder_Classifier(nn.Module):
	def __init__(self,classes):
		super().__init__()
		#self.z_dim=z_dim
		self.classes=classes
		self.encoder= MobileNet_Extractor()
		self.sigmoid=nn.Sigmoid()
		for param in self.encoder.parameters():
		  	param.requires_grad = False
		#encoder output shape=[1, 960, 1, 1]
		self.fc1=nn.Linear(1920,256)
		self.fc2=nn.Linear(256,128)
		self.fc3=nn.Linear(128,64)
		self.fc4=nn.Linear(64,32)
		self.fc5=nn.Linear(32,self.classes)
		self.relu=nn.ReLU()
	def forward(self,x):
		#neutral image
		neutral=x[:,0,:,:,:]
		#target image
		target_expression=x[:,1,:,:,:]
		neutral_emb=self.encoder(neutral)
		target_expression_emb=self.encoder(target_expression)
		neutral_emb=neutral_emb.view(-1,960)
		target_expression_emb=target_expression_emb.view(-1,960)
		difference_emb=target_expression_emb-neutral_emb
		#print("neutral:{}".format(neutral_emb))
		#print("target:{}".format(target_expression_emb))
		#print("difference:{}".format(torch.sum(difference_emb,1)))
		difference_emb=torch.cat((difference_emb,target_expression_emb),dim=1)
		out=self.relu(self.fc1(difference_emb))
		out=self.relu(self.fc2(out))
		out=self.relu(self.fc3(out))
		out=self.relu(self.fc4(out))
		out=self.fc5(out)
		return out