import torch
import torch.nn as nn
from torchvision import transforms
from pytorchcv.model_provider import get_model as ptcv_get_model

class AtkModel(nn.Module):
    def __init__(self, model_name, pretrained=False):
        super().__init__()
        self.model = ptcv_get_model(model_name, root='/tmp2/b07501122/.torch', pretrained=pretrained)
        self.normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

    def forward(self, x):
        nor_x = torch.zeros_like(x)
        for i in range(len(x)):
            nor_x[i] = self.normalize(x[i])
        outputs = self.model(nor_x)
        return outputs

class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     z_dim, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            # state size. (64*8) x 4 x 4
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            # state size. (64*4) x 8 x 8
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            # state size. (64*2) x 16 x 16
            nn.ConvTranspose2d(64 * 2,     64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(    64,      3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.shape[0], -1, 1, 1)
        output = self.main(z)
        output = (output + 1) / 2
        return output
