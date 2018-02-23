import torch
import torch.nn as nn
import numpy as np

class Spread_Loss(nn.Module):
    def __init__(self, end_epoch,decay_steps,init_margin = 0.2, max_margin = 0.9):
        super(Spread_Loss, self).__init__()        

        num_steps = end_epoch//decay_steps
        
        self.init_margin = init_margin
        self.max_margin = max_margin

        ratio = self.max_margin/self.init_margin
        self.decay_rate = ratio ** (1/float(num_steps))
        self.decay_steps = decay_steps
        self.margin = self.init_margin
        

    def forward(self,x,target,epoch_num):
        use_cuda = x.is_cuda
        
        # next(self.parameters()).is_cuda
        
        self.margin = self.init_margin * self.decay_rate **(epoch_num/self.decay_steps)
        self.margin = min(self.margin,self.max_margin)
        b = x.size(0)
        
        rows = torch.LongTensor(np.array(range(b)))
        if use_cuda:
            rows = rows.cuda()
        a_t = torch.diag(torch.index_select(x,1,target))
        

        a_t_stack = a_t.view(b,1).expand(b,x.size(1)).contiguous() #b,10

        u = self.margin-(a_t_stack-x) #b,10
        u = nn.functional.relu(u)**2
        u[rows,target]=0
        loss = torch.sum(u)/b
        
        return loss

        # self.step_curr += 1
        # for idx_param_group,param_group in enumerate(self.optimizer.param_groups): 
        #     # print idx_param_group,param_group['lr'],
        #     if self.init_lr[idx_param_group]!=0:
        #         new_lr = self.init_lr[idx_param_group] * self.decay_rate **(self.step_curr/self.decay_steps)
        #         new_lr = max(new_lr ,self.min_lr)
        #         param_group['lr'] = new_lr
            # print param_group['lr']