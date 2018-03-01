import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
    
class Spread_Loss(nn.Module):
    def __init__(self, end_epoch=100,decay_steps=5,init_margin = 0.2, max_margin = 0.9):
        super(Spread_Loss, self).__init__()        

        print end_epoch,decay_steps,init_margin, max_margin

        # end_epoch = int(end_epoch)
        # decay_steps = int(decay_steps)
        # init_margin = float(init_margin)
        # max_margin = float(max_margin)

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

class Spread_Loss_Multi(nn.Module):
    def __init__(self,end_epoch,decay_steps,init_margin = 0.2, max_margin = 0.9):
        super(Spread_Loss_Multi, self).__init__()        

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

        print 'target',target.size(),target.data
        print 'x',x.size(),x.data

        a_t_stack = torch.ones(b,x.size(1),x.size(1))
        if use_cuda:
            a_t_stack = a_t_stack.cuda()
        a_t_stack = Variable(a_t_stack)

        mask = 1-target.view(target.size(0),1,target.size(1))
        mul = target.view(target.size(0),target.size(1),1)
        a_t_stack = a_t_stack * mul

        pred_stack = x.view(x.size(0),1,x.size(1)).expand(x.size(0),x.size(1),x.size(1))

        a_t_stack = a_t_stack*x.view(x.size(0),x.size(1),1).expand(x.size(0),x.size(1),x.size(1))

        diff = self.margin-(a_t_stack - pred_stack)
        diff = diff*mul
        diff = diff*mask
        print diff
        diff = nn.functional.relu(diff)**2
        print diff
        loss = torch.sum(diff)/b


        # print pred_stack.size()
        # print a_t_stack.size()
        # print mask.size()
        # print diff
        
        return loss

def sanity_check(labels,predictions,m=0.1):
    a_t_stack = np.ones((labels.shape[0],labels.shape[1],labels.shape[1]))
    mul = labels[:,:,np.newaxis]
    mask = 1-labels[:,np.newaxis,:]
    a_t_stack = a_t_stack * mul
    # for dim in range(a_t_stack.shape[0]):

    pred_stack = np.tile(predictions[:,np.newaxis,:],(1,labels.shape[1],1))
    
    a_t_stack = a_t_stack*np.tile(predictions[:,:,np.newaxis],(1,1,labels.shape[1]))
    
    diff = m-(a_t_stack - pred_stack)
    
    diff = diff*mul
    
    diff = diff*mask
    print diff


def main():
    labels = np.array([[0,1,0,1],[1,0,0,0],[0,0,1,1]])
    predictions = np.random.rand(3,4)
    sanity_check(labels,predictions)

    labels = Variable(torch.Tensor(labels))
    output = Variable(torch.Tensor(predictions))
    criterion = Spread_Loss_Multi(50,5,max_margin=0.1)
    criterion(output,labels,1)



if __name__=='__main__':
    main()
