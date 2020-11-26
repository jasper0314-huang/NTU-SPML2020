import sys
sys.path.append("src")
from _model import AtkModel
from _utils import SimpleDataset
from attacks import mifgsm

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms

import time
import pickle
# config
##############################
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("model", type=str, help="training model (pytorchcv)")
parser.add_argument("--sep", type=int, default=1, help="epoch gap for regenerate adversarial examples")
parser.add_argument("--bs", type=int, default=128, help="batch size")
parser.add_argument("--lr", type=float, default=1e-1, help="learning rate")
parser.add_argument("--epoch", type=int, default=120, help="epoch")
parser.add_argument("--friendly", type=bool, default=False, help="wether using friendly setting")
parser.add_argument("--rand_start", type=bool, default=True, help="random initialize start point as PGD attack")
# mifgsm config
parser.add_argument("--eps", type=float, default=8/255, help="Linf constraint of IFGSM *default(8/255)")
parser.add_argument("--iter", type=int, default=10, help="iter of IFGSM")
parser.add_argument("--alpha", type=float, default=0.2, help="alaph of IFGSM")
parser.add_argument("--mu", type=float, default=0.0, help="mu of IFGSM")
###
parser.add_argument("--gpu", type=int, default=0, help="GPU device")
parser.add_argument("--log_path", type=str, default="log", help="log file save to {log_path}_{model}")
parser.add_argument("--save_path", type=str, default="ckpt", help="ckpt file save to {save_path}_{model}")

args = parser.parse_args()
if args.gpu != 0:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

device = "cuda" if torch.cuda.is_available() else "cpu"

# model
##############################
model = AtkModel(args.model, pretrained=True).to(device)

# optimizer / scheduler
##############################
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.1)

# criterion
criterion = nn.CrossEntropyLoss()

# attack method
attack_fn = mifgsm.MI_FGSM_batch(args.eps, args.alpha, args.mu, args.iter, args.friendly, args.rand_start)

# data
##############################

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.ToTensor()
])

cifar_train_dataset = torchvision.datasets.CIFAR10(
        root = "~/.torch",
        train = True,
        download = True,
        transform = train_transform)

cifar_dataset = cifar_train_dataset
cifar_loader = DataLoader(cifar_dataset, batch_size=args.bs, shuffle=True)

train_size = len(cifar_loader.dataset)

############
# for testing propse
# delete it if no needed
test_X = torch.load("~/cifar_adv_imgs")
test_Y = torch.load("~/cifar_labels")

test_dataset = TensorDataset(test_X, test_Y)
test_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False)
############

# train
##############################
log_list = []
best_acc = 0
for e in range(1, args.epoch + 1):
    start = time.clock()

    if (e - 1)%args.sep == 0:
        regen = True
        adv_box = []
    else: regen = False

    # training
    ###################################
    hit, loss_sum, total_num = 0, 0, 0
    model.train()
    for i, (x, y) in enumerate(cifar_loader):

        bs = x.shape[0]
        total_num += bs
        print(f"Batch: [{total_num} / {train_size}] | tmp_acc: {hit / total_num:.4f}", end="\r")

        # perturb
        if regen:
            model.eval()
            x, y = x.to(device), y.to(device)
            adv_x = attack_fn(x, y, [model])
            adv_box.append((adv_x, y.detach().cpu()))
            model.train()
        x, y = adv_box[i]

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
    model.eval()
    test_hit, test_loss_sum, test_num = 0, 0, 0
    for x, y in test_loader:
        bs = x.shape[0]
        test_num += bs

        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            output = model(x)
        loss = criterion(output, y)
        pred = torch.argmax(output, dim=1)

        # record
        test_loss_sum += loss.item() * bs
        test_hit += torch.sum(pred == y).item()
    
    test_loss = test_loss_sum / test_num
    test_acc = test_hit / test_num

    # log
    ##################################
    Min, sec = divmod(int(time.clock() - start), 60)
    log = f"Epoch: [{e} / {args.epoch}] ({Min} min {sec} sec) | Train loss: {train_loss:.4f} Acc: {train_acc:.4f} "
    log += f"| Test loss: {test_loss:.4f} Acc: {test_acc:.4f} "
    log += "*" if train_acc > best_acc else ""
    print(log)
    log_list.append((train_loss, train_acc, test_loss, test_acc))

    # save
    ##################################
    with open(f"{args.log_path}_{args.model}", "wb") as log_fp:
        pickle.dump(log_list, log_fp)
    torch.save(model.state_dict(), f"{args.save_path}_{args.model}")

    # scheduler
    ##################################
    scheduler.step()











