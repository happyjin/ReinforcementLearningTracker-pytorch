import torch
import torch.nn as nn
import torch.utils as utils
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.distributions import Normal
import numpy as np
import deepdish as dd
import matplotlib.pyplot as plt


# RNN module which apply LSTM
class RNN_module(nn.Module):
    def __init__(self, n_features, i_size, h_size, n_batch):
        """
        initial function for RNN module
        :param i_size: input size
        :param h_size: number of features in the hidden state h
        """
        super(RNN_module, self).__init__()
        self.n_batch = n_batch
        
        self.fc = nn.Linear(in_features=n_features, out_features=i_size)
        self.dropout = nn.Dropout(p=0.5)
        self.rnn = nn.LSTM(
            input_size=i_size,
            hidden_size=h_size, # The number of features in the hidden state h
            num_layers=1,       # Number of recurrent layers, the paper use one hidden layer LSTM, namely n_layers=1
            batch_first=True    # If True, then the input & output tensors are provided as (batch, seq, feature)
        )
        print('-----initializing RNN-----')

    def forward(self, combo, h_state, c_state):
        """
        forward pass function for RNN module
        :param input: input with shape (batch, time_step, input_size)
        :param h_state: hidden state with shape (n_layers, batch, hidden_size)
        :param c_state: cell state with shape (n_layers, batch, hidden_size)
        :return: hidden state and cell state for the next time step
        """
        input = self.fc(combo)
        input = self.dropout(input)
        #print(input.size())
        step = 1 # only process one frame at once
        input = input.view(self.n_batch, step, input.size(1))
        #print(input.size())
        output, (h_state, c_state) = self.rnn(input, (h_state, c_state))
        return h_state, c_state

    
def compute_iou(bbox1, bbox2):
    """
    Compute the intersection over union of two set of boxes, each box is [x1,y1,w,h]
    :param bbox1: (tensor) bounding boxes, size [N,4]
    :param bbox2: (tensor) bounding boxes, size [M,4]
    :return:
    """
    # compute [x1,y1,x2,y2] w.r.t. top left and bottom right coordinates separately
    b1x1y1 = bbox1[:,:2]-bbox1[:,2:]**2 # [N, (x1,y1)=2]
    b1x2y2 = bbox1[:,:2]+bbox1[:,2:]**2 # [N, (x2,y2)=2]
    b2x1y1 = bbox2[:,:2]-bbox2[:,2:]**2 # [M, (x1,y1)=2]
    b2x2y2 = bbox2[:,:2]+bbox2[:,2:]**2 # [M, (x1,y1)=2]
    box1 = torch.cat((b1x1y1.view(-1,2), b1x2y2.view(-1, 2)), dim=1) # [N,4], 4=[x1,y1,x2,y2]
    box2 = torch.cat((b2x1y1.view(-1,2), b2x2y2.view(-1, 2)), dim=1) # [M,4], 4=[x1,y1,x2,y2]
    N = box1.size(0)
    M = box2.size(0)

    tl = torch.max(
        box1[:,:2].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:,:2].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )
    br = torch.min(
        box1[:,2:].unsqueeze(1).expand(N,M,2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:,2:].unsqueeze(0).expand(N,M,2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = br - tl  # [N,M,2]
    #wh[(wh<0).detach()] = 0
    wh[wh<0] = 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:,2]-box1[:,0]) * (box1[:,3]-box1[:,1])  # [N,]
    area2 = (box2[:,2]-box2[:,0]) * (box2[:,3]-box2[:,1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


if __name__ == "__main__":
    # experiment for parameter update
    # set hyperparameters
    n_data = 20                   # 20 data needs two process
    n_features = 100              # YOLO output plus padding ground truth location or zeros
    hidden_size = input_size = 15 # input size is the size of input of RNN
    time_steps = 10               # a.k.a. N=10 in the paper
    n_batch = 1                   # number of batch
    n_layers = 1                  # number of recurrent layers
    N_samples = 3                 # number of sample locations for each time step
    n_epochs = 10                 # number of epochs for training
    sigma = torch.FloatTensor(4)  # variance for (x,y,w,h)
    sigma[:] = 0.01
    learning_rate = 1e-2

    #### load data ####
    gt = dd.io.load('../dataset/train_girl/normalized_bbox.h5')
    gt = Variable(torch.from_numpy(gt).float())
    combo = Variable(torch.randn((n_data, n_features))) # input is toy data for the experiment


    #### initialize FC module and RNN module ####
    #fc = FC(n_features, input_size)
    rnn = RNN_module(n_features, input_size, input_size, hidden_size)


    #### training process ####
    for epoch in range(n_epochs): # run the policy for n_epochs
        loop_begin = 0  # loop pointer
        print('')
        print('the epoch', epoch+1)
        # randomly generate initial hidden and cell states for each first of 10 frames
        h_state = Variable(torch.randn(n_layers, n_batch, hidden_size), requires_grad=True)
        c_state = Variable(torch.randn(n_layers, n_batch, hidden_size), requires_grad=True)
        gt_pointer = 0
        for loop in range(n_data // time_steps): # for each 10 frames
            #### forward pass ####
            # get input of RNN from FC layer output
            input = combo[loop_begin:loop_begin + time_steps]
            loop_begin += time_steps

            # compute mu from hidden states and sample N_samples location for each time step t
            sample_location = Variable(torch.FloatTensor(n_batch, time_steps, N_samples, 4))  # 4 is location (x,y,w,h)
            mu_tensor = Variable(torch.FloatTensor(time_steps, 4))
            for index_batch in range(n_batch):  # index for batch
                for t in range(time_steps):  # index for time_step
                    # for each time step apply RNN and get h_state, c_state for current time step
                    h_state, c_state = rnn(input, h_state, c_state)
                    # compute network output mean mu of location which contains (x,y,w,h)
                    mu = h_state[0, 0][-4:]  # [0,0] means the first recurrent layer and batch
                    mu_tensor[t] = mu
                    # randomly sample N location predictions for current time step
                    m = Normal(mu, sigma)
                    for index_sample in range(N_samples):  # sample n_samples from Gaussian distribution with mean mu and var sigma
                        # randomly sample predictions for N episodes, namely N_samples
                        sample_location[index_batch, t, index_sample, :] = m.sample().clone()

            # compute reward
            rep_gt = gt[gt_pointer:gt_pointer + time_steps].repeat(1, N_samples).view(1, time_steps, N_samples, 4)
            gt_pointer += time_steps
            abs_sub = (sample_location - rep_gt).abs()
            avg_val = abs_sub.mean(dim=-1)
            max_val = abs_sub.max(dim=-1)[0]
            reward = -avg_val - max_val

            # compute baseline
            baseline = reward.sum(dim=-1) / N_samples
            baseline_rep = baseline.view(-1, 1).repeat(1, N_samples)
            R_b = (reward - baseline_rep)  # (R^t_i - b_t) with shape=(N*T)
            G = reward.sum() # total reward or the expectation of total reward
            print('total reward:', G.data.numpy()[0])


            #### backward pass ####
            # get parameter size list of all parameter tensor
            size_list = []
            for param in rnn.parameters():
                size_list.append(list(param.size()))

            # compute dimension of each parameter vector
            param_size = 0
            truncate_size = []  # store size of each parameters
            for size in size_list:
                if len(size) == 2:
                    param_size = size[0] * size[1] + param_size
                    truncate_size.append(size[0] * size[1])
                else:
                    param_size += size[0]
                    truncate_size.append(size[0])
            truncate_size = np.array(truncate_size)

            # compute gradient of mu w.r.t. W
            gradient_mu = torch.FloatTensor(time_steps, 4, param_size)  ## it need to add batch_index as well
            for t in range(mu_tensor.size(0)):  # time_step
                for l in range(mu_tensor.size(1)):  # 4-dim location which contains (x,y,w,h)
                    mu_tensor[t, l].backward(retain_graph=True)
                    # compute grad. of fc layer
                    for index, param in enumerate(rnn.parameters()):
                        # print(param.grad.data.size())
                        if index == 0:
                            param_vector = param.grad.data.clone().view(-1)
                        else:
                            param_vector = torch.cat((param_vector, param.grad.data.clone().view(-1)))
                        param.grad.data.zero_()
                    gradient_mu[t, l, :] = param_vector

            # compute gradient of policy w.r.t. mu
            difference = sample_location - mu_tensor.repeat(1, N_samples).view(1, time_steps, N_samples, 4)
            gradient_policy = difference / sigma[0] ** 2

            # compute gradient of W using chain rule
            # [0,t] means batch index 0 at time step t
            gradient_W = torch.bmm(gradient_policy.data.squeeze(), gradient_mu)
            factor = R_b.squeeze().view(time_steps, N_samples, 1).repeat(1, 1, param_size).view(time_steps, N_samples,
                                                                                                param_size)
            # sum up gradient of W for N_sample at all time_steps
            gradient_G = gradient_W * factor.data

            # sum up gradient of W for N_sample at all time_steps
            gradient_sum = gradient_G.sum(0).sum(0) / N_samples

            #### update parameters ####
            # update parameters of fc model using gradient ascent
            pointer = 0  # indicate the i-th parameter
            begin = 0  # indicate beginning position
            for param in rnn.parameters():
                param.data += learning_rate * gradient_sum[begin:begin + truncate_size[pointer]].view(size_list[pointer])
                begin += truncate_size[pointer]
                pointer += 1


    #### save model ####
    torch.save(rnn, './DRLT.pkl')



