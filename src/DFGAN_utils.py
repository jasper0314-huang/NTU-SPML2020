import math
import torch
import torch.nn as nn
from _model import Generator

class DFGAN():
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = Generator().to(self.device)
        self.model.load_state_dict(
                torch.load("src/ckpt/GAN_ckpt"))
        self.z_dim = self.model.z_dim

    def clean_imgs(self, x, init_num=10, rec_iter=500):
        '''
        argument(s):
            x:          tensor (bs, 3, 32, 32) in scale [0, 1]
            init_num:   the number of random choosen z-vector
            rec_iter:   the iteration of reconstruction performed
        return(s):
            clean_x:    tensor (bs, 3, 32, 32) in scale [0, 1]
        '''
        bs = x.shape[0]
        lr = 999

        z = torch.randn((bs * init_num, self.z_dim)).to(self.device)
        z.requires_grad_(True)
        x = x.unsqueeze(dim=1)  # (bs, 1, 3, 32, 32)
        
        optimizer = torch.optim.SGD([z], lr=lr, momentum=0.7)
            
        def adjust_lr(optimizer, init_lr, ite, rec_iter):
            new_lr = init_lr * 0.1 ** (ite / int(math.ceil(0.8 * rec_iter)))
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr
        
        self.model.eval()
        for ite in range(rec_iter):
            
            fake_imgs = self.model(z) # (bs * init_num, 3, 32, 32)
            fake_imgs = fake_imgs.view(bs, init_num, 3, 32, 32) #    (bs, init_num, 3, 32, 32)
                                                                # x: (bs,        1, 3, 32, 32)
            each_loss = torch.mean(
                    ((fake_imgs - x) ** 2).view(bs, init_num, -1), dim=2) # (bs, init_num)

            loss = torch.sum(torch.mean(each_loss, dim=1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            adjust_lr(optimizer, lr, ite, rec_iter)

        # find the z* of each images by picking smallest rec. loss
        z_idx = torch.argmin(each_loss, dim=1) # (bs,)
        
        clean_imgs = torch.zeros((bs, 3, 32, 32)).to(self.device)
        for i in range(bs):
            clean_imgs[i] = fake_imgs[i, z_idx[i]].detach()
        return clean_imgs