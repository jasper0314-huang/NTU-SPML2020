import torch
import torch.nn as nn

class FGSM_batch():

    def __init__(self, eps=8/255):
        self.eps = eps
        self.criterion = nn.CrossEntropyLoss()

    def __call__(self, x, labels, models):
        '''
        x: FloatTensor (bs, 3, 32, 32) in [0, 1] scale
        labels: LongTensor (bs,)
        models: list of proxy model
        
        // Note that all parameters should in same device
        '''
        x_adv = x.clone().detach()
        x_adv.requires_grad_(True)
        
        factor = 1 / len(models)
        for i, model in enumerate(models):
            if i == 0: outputs = factor * model(x_adv)
            else: outputs += factor * model(x_adv)
        
        loss = self.criterion(outputs, labels)
        for model in models:
            model.zero_grad()
        loss.backward()
        
        x_adv = x_adv + self.eps * torch.sign(x_adv.grad)
        x_adv = torch.clamp(x_adv, 0, 1)
        x_adv = x_adv.detach()
        
        return x_adv
