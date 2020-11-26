import sys
sys.path.append("src")
from _model import AtkModel, Generator
from _utils import SimpleDataset
from DFGAN_utils import DFGAN

import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

import time

# config
##############################
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("model", type=str, help="test model name (pytorchcv model)")
# defense GAN config
parser.add_argument("--dfgan", type=bool, default=False, help="Apply Defense-GAN preprocessing or not")
parser.add_argument("--init_num", type=int, default=5, help="Defense-GAN config: init_num")
parser.add_argument("--rec_iter", type=int, default=200, help="Defense-GAN config: rec_iter")
########################
parser.add_argument("--ckpt", type=str, default="", help="checkpoint path of model / default as pytorchcv pretrained weights")
parser.add_argument("--imgs", type=str, default="/tmp2/b07501122/SPML/adv_examples/cifar_imgs", help="path to images data(tensor)")
parser.add_argument("--labels", type=str, default="/tmp2/b07501122/SPML/adv_examples/cifar_labels", help="path to labels data(tensor)")
parser.add_argument("--bs", type=int, default=64, help="training batch_size")
parser.add_argument("--gpu", type=int, default=0, help="gpu number")

args = parser.parse_args()
if args.gpu != 0:
	import os
	os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

device = "cuda" if torch.cuda.is_available() else "cpu"

# model
##############################
model = AtkModel(args.model, pretrained=(args.ckpt == "")).to(device)
if args.ckpt != "":
	model.load_state_dict(torch.load(args.ckpt))

if args.dfgan:
	defense_GAN = DFGAN()

# data
##############################
test_X = torch.load(args.imgs)
test_Y = torch.load(args.labels)

test_dataset = TensorDataset(test_X, test_Y)
test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

test_num = len(test_loader.dataset)

# test
##############################
model.eval()
hit, accu_num = 0, 0
for i, (x, y) in enumerate(test_loader):
	bs = x.shape[0]
	accu_num += bs
	print(f"{accu_num} / {test_num}", end="\r")

	x = x.to(device)
	y = y.to(device)
    
##############
	import torchvision
	if i == 0:
		torchvision.utils.save_image(x.detach().cpu(), 'tgt.png')
#############
	if args.dfgan:
		x = defense_GAN.clean_imgs(x, init_num=args.init_num, rec_iter=args.rec_iter)
############
	if i == 0:
		torchvision.utils.save_image(x.detach().cpu(), 'rec.png')
#############
	with torch.no_grad():
		output = model(x)
	pred = torch.argmax(output, dim=1)
	hit += torch.sum(pred == y).item()
print("=====================================")
print(f"model: {args.model}")
print(f"checkpoint: {args.ckpt}")
print(f"testing set: {args.imgs}")
print(f"Acc: {hit / accu_num:.4f}")
print("=====================================")
		












