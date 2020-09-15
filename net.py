import torch.nn as nn
import torch
from function import normal
from function import calc_mean_std
import scipy.stats as stats
from torchvision.utils import save_image
import random
decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU()  # relu5-4
)


class CA(nn.Module):
    def __init__(self, in_dim):
        super(CA, self).__init__()
        self.f = nn.Conv2d(in_dim , in_dim , (1,1))
        self.g = nn.Conv2d(in_dim , in_dim , (1,1))
        self.h = nn.Conv2d(in_dim , in_dim , (1,1))
        self.softmax  = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_dim, in_dim, (1, 1))
     
    def forward(self,content_feat,style_feat):
    
        B,C,H,W = content_feat.size()
        F_Fc_norm  = self.f(normal(content_feat)).view(B,-1,H*W).permute(0,2,1)
        
    

        B,C,H,W = style_feat.size()
        G_Fs_norm =  self.g(normal(style_feat)).view(B,-1,H*W) 
   
        energy =  torch.bmm(F_Fc_norm,G_Fs_norm)
        attention = self.softmax(energy)
        

        H_Fs = self.h(style_feat).view(B,-1,H*W)
        out = torch.bmm(H_Fs,attention.permute(0,2,1) )
        B,C,H,W = content_feat.size()
        out = out.view(B,C,H,W)
        out = self.out_conv(out)
     
        out += content_feat
        
        return out
 
class Style_SA(nn.Module):
    def __init__(self, in_dim):
        super(Style_SA, self).__init__()
        self.f = nn.Conv2d(in_dim , in_dim , (1,1))
        self.g = nn.Conv2d(in_dim , in_dim , (1,1))
        self.h = nn.Conv2d(in_dim , in_dim , (1,1))
        self.softmax  = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_dim, in_dim, (1, 1))

    def forward(self,style_feat):

        B,C,H,W = style_feat.size()
        F_Fc_norm  = self.f(style_feat).view(B,-1,H*W)
        
   
        B,C,H,W = style_feat.size()
        G_Fs_norm =  self.g(style_feat).view(B,-1,H*W).permute(0,2,1) 

        energy =  torch.bmm(F_Fc_norm,G_Fs_norm)
        attention = self.softmax(energy)


        H_Fs = self.h(normal(style_feat)).view(B,-1,H*W)
        out = torch.bmm(attention.permute(0,2,1), H_Fs)
        
        out = out.view(B,C,H,W)
        out = self.out_conv(out)
        out += style_feat
        return out
class Content_SA(nn.Module):
    def __init__(self, in_dim):
        super(Content_SA, self).__init__()
        self.f = nn.Conv2d(in_dim , in_dim , (1,1))
        self.g = nn.Conv2d(in_dim , in_dim , (1,1))
        self.h = nn.Conv2d(in_dim , in_dim , (1,1))
        self.softmax  = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_dim, in_dim, (1, 1))

    
    def forward(self,content_feat):

        B,C,H,W = content_feat.size()
        F_Fc_norm  = self.f(normal(content_feat)).view(B,-1,H*W).permute(0,2,1)


        B,C,H,W = content_feat.size()
        G_Fs_norm =  self.g(normal(content_feat)).view(B,-1,H*W) 

        energy =  torch.bmm(F_Fc_norm,G_Fs_norm)
        attention = self.softmax(energy)
        
        H_Fs = self.h(content_feat).view(B,-1,H*W)
        out = torch.bmm(H_Fs,attention.permute(0,2,1) )
        B,C,H,W = content_feat.size()
        out = out.view(B,C,H,W)
        out = self.out_conv(out)
        out += content_feat
  
        return out

class Multi_Adaptation_Module(nn.Module):
    def __init__(self, in_dim):
        super(Multi_Adaptation_Module, self).__init__()

        self.CA=CA(in_dim)
        self.CSA=Content_SA(in_dim)
        self.SSA=Style_SA(in_dim)

    def forward(self, content_feats, style_feats):
      
        content_feat = self.CSA(content_feats[-2])
        style_feat = self.SSA(style_feats[-2])
        Fcsc = self.CA(content_feat, style_feat)
      
        return Fcsc

class Net(nn.Module):
    def __init__(self, encoder, decoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1
        #transform
        self.ma_module = Multi_Adaptation_Module(512)
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4', 'enc_5']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

    # extract relu1_1, relu2_1, relu3_1, relu4_1, relu5_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

           
    def forward(self, content, content1, style, style1):
        #print(content.size())
        style_feats = self.encode_with_intermediate(style)
        content_feats = self.encode_with_intermediate(content)
        style_feats1 = self.encode_with_intermediate(style1)
        content_feats1 = self.encode_with_intermediate(content1)


        Ics = self.decoder(self.ma_module(content_feats, style_feats))
        Ics_feats = self.encode_with_intermediate(Ics)
        # Content loss
   
        Ics1 = self.decoder(self.ma_module(content_feats, style_feats1))
        Ics1_feats = self.encode_with_intermediate(Ics1)
        Ic1s = self.decoder(self.ma_module(content_feats1, style_feats))
        Ic1s_feats = self.encode_with_intermediate(Ic1s)
        
        #Identity losses lambda 1
        Icc = self.decoder(self.ma_module(content_feats, content_feats))
        Iss = self.decoder(self.ma_module(style_feats, style_feats)) 
    
        #Identity losses lambda 2
        Icc_feats=self.encode_with_intermediate(Icc)
        Iss_feats=self.encode_with_intermediate(Iss)
        return style_feats, content_feats, style_feats1, content_feats1 ,Ics_feats,Ics1_feats,Ic1s_feats,Icc,Iss,Icc_feats,Iss_feats
  

