import sys
sys.path.append("src")
from _model import AtkModel
from _utils import SimpleDataset

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

import time
# config
##############################
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--bs", type=int, default=64, help="training batch_size")
parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
parser.add_argument("--epoch", type=int, default=100, help="training Epoch")
parser.add_argument("--gpu", type=int, default=0, help="gpu number")
parser.add_argument("--model", type=str, default="resnet1001_cifar10", help="pytorchcv model name")
parser.add_argument("--pretrained", type=bool, default=True, help="use pytorchcv pretrained weights")
parser.add_argument("--save_path", type=str, default="/tmp2/b07501122/adv_ckpt/ckpt_of", help="checkpoint save to {save_path}_{model}")

args = parser.parse_args()
if args.gpu != 0:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

device = "cuda" if torch.cuda.is_available() else "cpu"

# model
##############################
model = AtkModel(args.model, pretrained=args.pretrained).to(device)

# optimizer
##############################
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# criterion
criterion = nn.CrossEntropyLoss()

# data
##############################
adv_imgs = torch.load("/tmp2/b07501122/SPML/adv_examples/cifar_adv_imgs_02_whole")
labels = torch.load("/tmp2/b07501122/SPML/adv_examples/cifar_labels_whole")
print(adv_imgs.shape)   # should be FloatTensor with shape (60000, 3, 32, 32) (in scale [0, 1])
print(labels.shape)     # should be LongTensor with shape (60000,)

train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.ToTensor()
])

train_dataset = SimpleDataset(adv_imgs[:50000], labels[:50000], train_transform)
train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)

test_dataset = SimpleDataset(adv_imgs[50000:], labels[50000:])
test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)

train_size = len(train_loader.dataset)

# train
##############################
best_acc = 0
for e in range(1, args.epoch + 1):
    start = time.clock()
    # training
    ###################################
    hit, loss_sum, total_num = 0, 0, 0
    model.train()
    for x, y in train_loader:
        bs = x.shape[0]
        total_num += bs
        print(f"Batch: [{total_num} / {train_size}] | tmp_acc: {hit / total_num:.4f}", end="\r")    

        # update
        x, y = x.to(device), y.to(device)
        o = model(x)
        loss = criterion(o, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # record
        loss_sum += loss.item() * bs
        hit += torch.sum(torch.argmax(o, dim=1) == y).item()

    train_loss = loss_sum / total_num
    train_acc = hit / total_num
    # testing
    ###################################
    hit, loss_sum, total_num = 0, 0, 0
    model.eval()
    for x, y in test_loader:
        bs = x.shape[0]
        total_num += bs

        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            o = model(x)
            loss = criterion(o, y)
        loss_sum += loss.item() * bs
        hit += torch.sum(torch.argmax(o, dim=1) == y).item()

    test_loss = loss_sum / total_num
    test_acc = hit / total_num
    # log
    ###################################
    Min, sec = divmod(int(time.clock() - start), 60)
    log = f"Epoch: [{e} / {args.epoch}] ({Min} min {sec} sec) | Train loss: {train_loss:.4f} Acc: {train_acc:.4f} "
    log += f"| Test loss: {test_loss:.4f} Acc: {test_acc:.4f}"
    log += " *" if train_acc > best_acc else ""
    print(log)

    # save
    ##################################
    if train_acc > best_acc:
        best_acc = train_acc
        torch.save(model.state_dict(), f"{args.save_path}_{args.model}")

        












