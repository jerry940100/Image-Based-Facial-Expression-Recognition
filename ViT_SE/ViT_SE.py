from transformers import ViTFeatureExtractor, ViTModel
import torch
import torch.nn as nn
class SE_block(nn.Module):
    def __init__(self):
        super().__init__()
        self.excitation=nn.Sequential(
            nn.Linear(768,192),
            nn.ReLU(),
            nn.Linear(192,768),
            nn.ReLU()
        )
    def forward(self,x):
        res=x
        out=self.excitation(x)
        out=res*out
        return out
#把SE_block加進去
class ViT_SE(nn.Module):
    def __init__(self):
        super().__init__()
        self.ViT_Transformer=ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.SE_Block=SE_block()
        self.linear=nn.Linear(768,7)
    def forward(self,x):
        out=self.ViT_Transformer(x)
        out=out.pooler_output#[Batch_size,768]
        Excitation_out=self.SE_Block(out)
        Final_out=self.linear1(Excitation_out)
        return Final_out