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
        self.layers = nn.ModuleList(features)
    def forward(self,x):
        for ii,model in enumerate(self.layers):
            x = model(x)
        return x

#import torchvision.models.resnet50 as ResNet
from torchvision.models import resnet50 as ResNet
def resnet50(weights_path=None, **kwargs):
    """Constructs a ResNet-50 model.
    """
    model = ResNet(False, [3, 4, 6, 3], **kwargs)
    if weights_path:
        import pickle
        with open(weights_path, 'rb') as f:
            obj = f.read()
        weights = {key: torch.from_numpy(arr) for key, arr in pickle.loads(obj, encoding='latin1').items()}
        model.load_state_dict(weights)
    return model


class ResNet50_Extractor(nn.Module):
    def __init__(self):
        super().__init__()
        #download the resnet50 pretrained weight https://drive.google.com/open?id=1A94PAAnwk6L7hXdBXLFosB_s0SzEhAFU
        #put pretrained file in this folder
        self.resnet50 = resnet50("./resnet50_ft_weight.pkl", num_classes=8631)  # Pretrained weights fc layer has 8631 outputs
        del self.resnet50.fc
        self.feature=nn.Sequential(
            self.resnet50.conv1,
            self.resnet50.bn1,
            self.resnet50.relu,
            self.resnet50.layer1,
            self.resnet50.layer2,
            self.resnet50.layer3,
            self.resnet50.layer4,
            self.resnet50.avgpool#2048 dim
        )
    def forward(self,x):
        x=self.feature(x)
        return x

class Pretrained_Encoder_Classifier(nn.Module):
    def __init__(self,classes):
        super().__init__()
        #self.z_dim=z_dim
        self.classes=classes
        self.encoder_emotion=MobileNet_Extractor()
        self.encoder_id=ResNet50_Extractor()
        for param in self.encoder_id.parameters():
          param.requires_grad = False
        #encoder output shape=[1, 960, 1, 1]
        self.fc1=nn.Linear(960+2048,256)
        self.fc2=nn.Linear(256,128)
        self.fc3=nn.Linear(128,64)
        self.fc4=nn.Linear(64,32)
        self.fc5=nn.Linear(32,self.classes)
        self.relu=nn.ReLU()
    def forward(self,x):
        #input images[batch_size,C,H,W]
        target=x
        emotion_emb=self.encoder_emotion(target)
        #print(emotion_emb.shape)
        id_emb=self.encoder_id(target)
        #print(id_emb.shape)
        emotion_emb=emotion_emb.view(-1,960)
        id_emb=id_emb.view(-1,2048)
        target_emb=torch.cat((emotion_emb,id_emb),dim=1)
        del id_emb,emotion_emb
        #print(difference_emb.shape)
        out=self.relu(self.fc1(target_emb))
        out=self.relu(self.fc2(out))
        out=self.relu(self.fc3(out))
        out=self.relu(self.fc4(out))
        out=self.fc5(out)
        return out