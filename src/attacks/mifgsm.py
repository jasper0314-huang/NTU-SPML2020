import torch
import torch.nn as nn

class MI_FGSM_batch():

    def __init__(self, eps=8/255, alpha=0.2, mu=1.0, ite=10, friendly=False, rand_start=False):
        self.eps = eps
        self.alpha = alpha
        self.mu = mu
        self.ite = ite
        self.criterion = nn.CrossEntropyLoss()
        self.friendly = friendly
        self.rand_start = rand_start

    def __call__(self, x, labels, models):
        '''
        x: FloatTensor (bs, 3, 32, 32) in [0, 1] scale
        labels: LongTensor (bs,)
        models: list of proxy model
        
        // Note that all parameters should in same device
        '''
        x_adv = x.clone().detach()              # clone 到 x_adv 以確保不會改到 x
        g = torch.zeros_like(x)                 # accumulated gradient terms

        # random start from a neighbor point: PGD attack
        if self.rand_start:
            x_adv += self.eps * (2 * torch.rand_like(x_adv) - 1)
        
        stop_flag = False # fixed if unfriendly
        for i in range(self.ite):
            
            x_adv.requires_grad_(True)
            
            # forward over all models
            factor = 1 / len(models)
            for j, model in enumerate(models):
                if j == 0: outputs = factor * model(x_adv)
                else: outputs += factor * model(x_adv)
            
            # loss and backward
            loss = self.criterion(outputs, labels)
            for model in models:
                model.zero_grad()
            loss.backward()

            # update by following equation:
            # g_t+1 = mu * g_t + grad / |grad|_1
            grad = x_adv.grad
            grad_norm = torch.norm(grad.reshape((grad.shape[0], -1)), p=1., dim=1)
            g = self.mu * g + grad / grad_norm.view((-1, 1, 1, 1))
            
            # early stopping or not
            # check if labels flipping
            if self.friendly:
                tgt_idx = (torch.argmax(outputs, dim=1) == labels)
                stop_flag = (torch.sum(tgt_idx).item() == 0)
                g[~tgt_idx] = 0

            # x_t+1 = x + alpha * sign(gt+1)
            x_adv = x_adv + self.alpha * self.eps * torch.sign(g)
            x_adv = torch.where(x_adv > x+self.eps, x+self.eps, x_adv)
            x_adv = torch.where(x_adv < x-self.eps, x-self.eps, x_adv)
            x_adv = torch.clamp(x_adv, 0, 1)
            x_adv = x_adv.detach()

            if stop_flag:
                break
        
        return x_adv.cpu()
