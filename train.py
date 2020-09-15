import importlib
import sys
importlib.reload(sys)
from function import normal
import argparse
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from function import calc_mean_std
from PIL import ImageFile
# from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import net 
import random
from sampler import InfiniteSamplerWrapper
#from torch.utils.data.sampler import RandomSampler
from torchvision.utils import save_image
cudnn.benchmark = True
Image.MAX_IMAGE_PIXELS = None  # Disable DecompressionBombError
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Disable OSError: image file is truncated


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)



class FlatFolderDataset(data.Dataset):
    def __init__(self, root, transform):
        super(FlatFolderDataset, self).__init__()
        self.root = root
        print(self.root)
        self.path = os.listdir(self.root)
        if os.path.isdir(os.path.join(self.root,self.path[0])):
            self.paths = []
            for file_name in os.listdir(self.root):
                for file_name1 in os.listdir(os.path.join(self.root,file_name)):
                    self.paths.append(self.root+"/"+file_name+"/"+file_name1)  
             
                     
        else:

            self.paths = list(Path(self.root).glob('*'))
        #print(self.paths)
        self.transform = transform

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(str(path)).convert('RGB')
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.paths)

    def name(self):
        return 'FlatFolderDataset'
def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr / (1.0 + args.lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', type=str, required=True,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', type=str, required=True,
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='model/vgg_normalised.pth')

# training options
parser.add_argument('--save_dir', default='./experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='./logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_decay', type=float, default=5e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=5.0)
parser.add_argument('--content_weight', type=float, default=1.0)
parser.add_argument('--n_threads', type=int, default=16)
parser.add_argument('--save_model_interval', type=int, default=10000)
args = parser.parse_args()

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
# writer = SummaryWriter(log_dir=args.log_dir)

decoder = net.decoder
#decoder = net.Decoder(net.decoder)
vgg = net.vgg

vgg.load_state_dict(torch.load(args.vgg))
vgg = nn.Sequential(*list(vgg.children())[:44])
with torch.no_grad():
    network = net.Net(vgg, decoder)
network.train()
network.to(device)
network = nn.DataParallel(network, device_ids=[0,1])

content_tf = train_transform()
style_tf = train_transform()

content_dataset = FlatFolderDataset(args.content_dir, content_tf)

# paths = []
# for i in os.listdir(args.style_dir):
#     for j in os.listdir(args.style_dir+"/"+i):
#         paths.append(args.style_dir+"/"+i+"/"+j) 
#         #print(paths)
style_dataset = FlatFolderDataset(args.style_dir, style_tf)

content_iter = iter(data.DataLoader(
    content_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter = iter(data.DataLoader(
    style_dataset, batch_size=args.batch_size,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))
content_iter1 = iter(data.DataLoader(
    content_dataset, batch_size=1,
    sampler=InfiniteSamplerWrapper(content_dataset),
    num_workers=args.n_threads))
style_iter1 = iter(data.DataLoader(
    style_dataset, batch_size=1,
    sampler=InfiniteSamplerWrapper(style_dataset),
    num_workers=args.n_threads))
optimizer = torch.optim.Adam([
                              {'params': network.module.decoder.parameters()},
                              {'params': network.module.ma_module.parameters()}], lr=args.lr)
mse_loss = nn.MSELoss()
def calc_content_loss( input, target):
      assert (input.size() == target.size())
      #assert (target.requires_grad is False)
      return mse_loss(input, target)

def calc_style_loss( input, target):
    assert (input.size() == target.size())
    #assert (target.requires_grad is False)
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return mse_loss(input_mean, target_mean) + \
           mse_loss(input_std, target_std)
def shuffle( feat):
    B,C,W,H = feat.size()
    x = [i for i in range(B)]
    random.shuffle(x)
    new_feat = feat[x[0]].view(1,C,W,H)
    for i in x[1:]:
        new_feat = torch.cat((new_feat,feat[i].view(1,C,W,H)),0)
    #print(new_feat.size())
    return new_feat

for i in tqdm(range(args.max_iter)):
    adjust_learning_rate(optimizer, iteration_count=i)
    content_images = next(content_iter).to(device)
    style_images = next(style_iter).to(device)
    B,C,W,H = content_images.size()
    content_images1 = next(content_iter).to(device)
    style_images1 = next(style_iter).to(device)

    content_images1 = content_images1.expand(B,C,W,H)
    style_images1 = style_images1.expand(B,C,W,H)

    style_feats, content_feats, style_feats1, content_feats1 ,Ics_feats,Ics1_feats,Ic1s_feats,Icc,Iss,Icc_feats,Iss_feats = network(content_images, content_images1,style_images,style_images1)
    
    loss_c = calc_content_loss(normal(Ics_feats[-1]), normal(content_feats[-1]))+calc_content_loss(normal(Ics_feats[-2]), normal(content_feats[-2]))
        # Style loss
    loss_s = calc_style_loss(Ics_feats[0], style_feats[0])
    for j in range(1, 5):
        loss_s += calc_style_loss(Ics_feats[j], style_feats[j])
    
    dis_loss_c = calc_content_loss(normal(shuffle(Ic1s_feats[-1])), normal(shuffle(Ic1s_feats[-1])))+calc_content_loss(normal(shuffle(Ic1s_feats[-2])), normal(shuffle(Ic1s_feats[-2]))) 
    dis_loss_s = calc_style_loss(shuffle(Ics1_feats[0]), shuffle(Ics1_feats[0]))
    for j in range(1, 5):
        dis_loss_s += calc_style_loss(shuffle(Ics1_feats[j]), shuffle(Ics1_feats[j]))
 
    l_identity1 = calc_content_loss(Icc,content_images)+calc_content_loss(Iss,style_images)

    l_identity2 = calc_content_loss(Icc_feats[0], content_feats[0])+calc_content_loss(Iss_feats[0], style_feats[0])
    for j in range(1, 5):
        l_identity2 += calc_content_loss(Icc_feats[j], content_feats[j])+calc_content_loss(Iss_feats[j], style_feats[j])



    loss_c = args.content_weight * loss_c
    loss_s = args.style_weight * loss_s
    if (loss_c + loss_s) < 10.0:
        k1 = 1
        k2 = 1
    else:
        k1=0
        k2=0
    loss = loss_c + loss_s + (dis_loss_c * k1) + (dis_loss_s * k2) + (l_identity1 * 50) + (l_identity2 * 1)

    print(loss.cpu().detach().numpy(),"content:",loss_c.cpu().detach().numpy(),"style:",loss_s.cpu().detach().numpy()
              ,"dc:",dis_loss_c.cpu().detach().numpy(),"ds:",dis_loss_s.cpu().detach().numpy()
              ,"l1:",l_identity1.cpu().detach().numpy(),"l2:",l_identity2.cpu().detach().numpy()
              )


    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # writer.add_scalar('loss_content', loss_c.item(), i + 1)
    # writer.add_scalar('loss_style', loss_s.item(), i + 1)
    # # writer.add_scalar('loss_identity1', l_identity1.item(), i + 1)
    # # writer.add_scalar('loss_identity2', l_identity2.item(), i + 1)
    # writer.add_scalar('total_loss', loss.item(), i + 1)    

    if (i + 1) % args.save_model_interval == 0 or (i + 1) == args.max_iter:
        print("-------------------")
        state_dict = network.module.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/decoder_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
        state_dict = network.module.ma_module.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict,
                   '{:s}/ma_module_iter_{:d}.pth'.format(args.save_dir,
                                                           i + 1))
# writer.close()
#torch.cuda.empty_cache()
