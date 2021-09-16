import torch
import torch.nn as nn
from Pretrained_Emotion_Encoder import Pretrained_Encoder_Classifier
#Pretrained Emotion Enocoder classes
#classes=(pretrain model類別數量)
pretrained=Pretrained_Encoder_Classifier(classes="Adjust by your pretrained classes")
#Load the Weight of the pretrained model
pretrained.load_state_dict(torch.load(PATH))#You have to adjust the PATH
pretrained.eval()

class Encoder_Classifier(nn.Module):
    def __init__(self,classes):
        super().__init__()
        self.classes=classes
        self.encoder=pretrained.encoder_emotion
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
        difference_emb=torch.cat((difference_emb,target_expression_emb),dim=1)
        out=self.relu(self.fc1(difference_emb))
        out=self.relu(self.fc2(out))
        out=self.relu(self.fc3(out))
        out=self.relu(self.fc4(out))
        out=self.fc5(out)
        return out


