import sys
sys.path.append("./src")
from _model import AtkModel
from attacks import fgsm, mifgsm

import tqdm
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

# config
##############################
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

parser.add_argument("--bs", type=int, default=16, help="batch size")
parser.add_argument("--eps", type=float, default=8/255, help="Linf constraint for attack")
parser.add_argument("--attack", type=str, default="mifgsm", help="attacks mehtods now supported: {fgsm/mifgsm}")
# mifgsm config
parser.add_argument("--iter", type=int, default=10, help="IFGSM iter")
parser.add_argument("--alpha", type=float, default=0.1, help="IFGSM alaph")
parser.add_argument("--mu", type=float, default=1.0, help="IFGSM mu")
parser.add_argument("--rand_start", type=bool, default=False, help="random initialize as PGD")
###
parser.add_argument("--gpu", type=int, default=0, help="gpu number")
parser.add_argument("--model", type=str, default="", help="pytorchcv model")
parser.add_argument("--model_ckpt", type=str, default="", help="model's checkpoint")
parser.add_argument("--models_file", type=str, default="", help="pytorchcv models file (ignore if model not empty)")
parser.add_argument("--save_file", type=str, default="/tmp2/b07501122/SPML/adv_examples/cifar_adv_imgs", help="adv images save path")

args = parser.parse_args()
if args.gpu != 0:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.gpu}"

device = "cuda" if torch.cuda.is_available() else "cpu"

# attack method
##############################
attacks = {
    'fgsm': fgsm.FGSM_batch(args.eps),
    'mifgsm': mifgsm.MI_FGSM_batch(args.eps, args.alpha, args.mu, args.iter, False, args.rand_start)
}
attack_fn = attacks[args.attack]

# dataset
##############################
cifar_test_dataset = torchvision.datasets.CIFAR10(
        root = "/tmp2/b07501122/.torch", 
        train = False, 
        download = True,
        transform = transforms.ToTensor())
'''
cifar_train_dataset = torchvision.datasets.CIFAR10(
        root = "/tmp2/b07501122/.torch",
        train = True,
        download = True,
        transform = transforms.ToTensor())
'''
cifar_dataset = cifar_test_dataset
cifar_loader = DataLoader(cifar_dataset, batch_size=args.bs, shuffle=False)

# model
##############################
models = []
if args.model != "":
    model = AtkModel(args.model, pretrained=(args.model_ckpt=="")).to(device)
    if args.model_ckpt != "":
        print(model.load_state_dict(torch.load(args.model_ckpt)))
    model.eval()
    models.append(model)
elif args.models_file != "":
    with open(args.models_file) as f:
        target_names = [line.strip() for line in f]
    
    for name in target_names:
        model = AtkModel(name, pretrained=True).to(device)
        model.eval()
        models.append(model)
else:
    sys.exit("must specify --model or --models_file option.")

test_model = AtkModel('resnet1001_cifar10', pretrained=True).to(device)
test_model.eval()

# attack
###############################
hit, adv_hit = 0, 0
total_num = 0
for i, d in enumerate(tqdm.tqdm(cifar_loader)):
    bs = len(d[0])
    total_num += bs

    imgs = d[0].to(device)
    labels = d[1].to(device)
    adv_imgs = attack_fn(imgs, labels, models).to(device)

    # testing
    with torch.no_grad():
        logits = test_model(imgs)
    preds = torch.argmax(logits, dim=1)
    hit += torch.sum(preds == labels).item()

    # adv testing
    with torch.no_grad():
        adv_logits = test_model(adv_imgs)
    adv_preds = torch.argmax(adv_logits, dim=1)
    adv_hit += torch.sum(adv_preds == labels).item()

    adv_imgs = adv_imgs.detach().cpu()
    if i == 0: adv_images = adv_imgs
    else: adv_images = torch.cat((adv_images, adv_imgs), dim=0)

print(f"\nAcc: {hit / total_num:.4f} | adv_Acc: {adv_hit / total_num:.4f}")
print("* testing on resnet1001_cifar10")
torchvision.utils.save_image(adv_images[:32], "check_adv_imgs.png")
torch.save(adv_images, args.save_file)
