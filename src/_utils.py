import torch
import torchvision
import numpy as np
#import imgaug.augmenters as iaa
from torch.utils.data import Dataset

class SimpleDataset(Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform is not None:
            x = self.transform(x)
        y = self.Y[idx]
        return x, y

def jpeg_compression_batch(x, quality, device):
    # x: (bs, 3, 32, 32)
    np_x = (x.detach().cpu()*255).permute(0, 2, 3, 1).numpy()
    np_x = np_x.astype(np.uint8)
    aug = iaa.JpegCompression(compression=100-quality)
    aug_np_x = aug(images=np_x)
    tr = torchvision.transforms.ToTensor()
    for i, lx in enumerate(aug_np_x):
        if i == 0: aug_x = tr(lx).unsqueeze(0)
        else: aug_x = torch.cat((aug_x, tr(lx).unsqueeze(0)), dim=0)
    return aug_x.to(device)
