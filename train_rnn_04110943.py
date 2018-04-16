
# coding: utf-8

# In[1]:


import os
from RNN import *


# ### use gpu

# In[2]:


os.environ["CUDA_VISIBLE_DEVICES"] = "4"
use_gpu = torch.cuda.is_available()


# ### speed up and save GPU memory

# In[3]:


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


# ### load dataset, namely feature+location a.k.a. combo

# In[4]:


combo_all = dd.io.load('20classes_combo_padzero_1000features.h5') # utilize pad ground truth


# In[5]:


gt_all = dd.io.load('20classes_normal_bboxes_all_sqrtwh_list.h5')


# In[6]:


len(gt_all)


# In[7]:


len(combo_all)


# In[8]:


combo_all[0].size()


# In[9]:


gt_all[0][0]


# ### set hyperparameters

# In[10]:


#n_data = 50                   # 20 data needs two process
n_features = combo_all[0].size(1)              # YOLO output plus padding ground truth location or zeros
hidden_size = input_size = 1004 # input size is the size of input of RNN
time_steps = 10               # a.k.a. N=10 in the paper
n_batch = 1                   # number of batch
n_layers = 1                  # number of recurrent layers
N_samples = 3                 # number of sample locations for each time step
n_epochs = 300                 # number of epochs for training
sigma = torch.FloatTensor(4).cuda()  # variance for (x,y,w,h)
sigma[:] = 0.05
learning_rate = 0.0007 #0.0006
n_dataset = 10


# ### save reward result

# In[11]:


#result_reward = np.zeros((n_epochs, n_data // time_steps))


# ### initialize FC module and RNN module

# In[12]:


rnn = RNN_module(n_features, input_size, hidden_size, n_batch).cuda() # run the module on GPU


# ### initialize FC layer

# In[13]:


for m in rnn.modules():
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


# ### set the model in "training mode"

# In[ ]:


rnn.train()


# ### training process

# In[ ]:


reward_record = []
reward_2nd_record = []
for epoch in range(n_epochs): # run the policy for n_epochs
    for ith_dataset in range(n_dataset):
        n_subtrain = gt_all[ith_dataset].shape[0] // 3 // time_steps
        n_subtrain *= time_steps
        combo_train = combo_all[ith_dataset][:n_subtrain]
        gt_train = gt_all[ith_dataset][:n_subtrain]
        #gt = Variable(torch.from_numpy(gt_train).float(), volatile=True) # run this variable on GPU
        #combo = Variable(combo_train).cuda() # run this variable on GPU
        
        loop_begin = 0  # loop pointer
        gt_pointer = 0  # ground truth array pointer
        #print('')
        print('epoch %d, the dataset %d' %(epoch+1, ith_dataset+1))
        # randomly generate initial hidden and cell states at begining of each epoch
        h_state = Variable(torch.randn(n_layers, n_batch, hidden_size)).cuda()
        c_state = Variable(torch.randn(n_layers, n_batch, hidden_size)).cuda()
        for loop in range(n_subtrain // time_steps): # for each 10 frames
            #### forward pass ####
            # get input of RNN from FC layer output
            input = combo_train[loop_begin:loop_begin + time_steps]
            input = Variable(input).cuda()
            #print(input.size())
            
            loop_begin += time_steps

            # compute mu from hidden states and sample N_samples location for each time step t
            sample_location = Variable(torch.FloatTensor(n_batch, time_steps, N_samples, 4), volatile=True).cuda()  # 4 is location (x,y,w,h)
            mu_tensor = Variable(torch.FloatTensor(time_steps, 4)).cuda()
            for index_batch in range(n_batch):  # index for batch
                for t in range(time_steps):  # index for time_step
                    # for each time step apply RNN and get h_state, c_state for current time step
                    #one_input = input[t,:].view(1,-1)
                    #one_input = Variable(one_input).cuda()
                    h_state, c_state = rnn(input[t,:].view(1,-1), h_state, c_state)
                    # compute network output mean mu of location which contains (x,y,w,h)
                    mu = h_state[0, 0][-4:]  # [0,0] means the first recurrent layer and batch
                    mu_tensor[t] = mu
                    # randomly sample N location predictions for current time step
                    m = Normal(mu, sigma)
                    for index_sample in range(N_samples):  # sample n_samples from Gaussian distribution with mean mu and var sigma
                        # randomly sample predictions for N episodes, namely N_samples
                        sample_location[index_batch, t, index_sample, :] = m.sample().clone()
                    # delete mu and m at the end of each loop in order to save memory

            # compute the first reward
            one_gt = gt_train[gt_pointer:gt_pointer + time_steps]
            gt = Variable(torch.from_numpy(one_gt).float(), volatile=True)
            gt = gt.cuda()
            rep_gt = gt.repeat(1, N_samples).view(1, time_steps, N_samples, 4)
            gt_pointer += time_steps
            abs_sub = (sample_location - rep_gt).abs()
            avg_val = abs_sub.mean(dim=-1)
            max_val = abs_sub.max(dim=-1)[0]
            reward = -avg_val - max_val
            G_display = reward.sum().cpu().data.numpy()[0]
            reward_record.append(G_display)
            print('1st total reward:', G_display)
            
            # compute baseline
            baseline = reward.sum(dim=-1) / time_steps
            #baseline = Variable(baseline.data, volatile=True).cuda()
            baseline_rep = baseline.view(-1,1).repeat(1,N_samples)
            R_b = (reward - baseline_rep) # (R^t_i - b_t) with shape=(N*T)
            
            '''
            # compute the second reward
            #box1_gt = gt_train[gt_pointer:gt_pointer + time_steps]  # (10, 4)
            reward_2 = Variable(torch.zeros(time_steps, N_samples)).cuda()
            for k in range(N_samples):
                box2_pred = sample_location.squeeze()[:, k, :]
                iou = compute_iou(gt, box2_pred)
                reward_2[:, k] = iou.diag()
            # print(reward_2.size())
            G = reward_2.sum()  # total reward or the expectation of total reward
            total_reward_2 = G.cpu().data.numpy()[0]
            if total_reward_2 != 0:
                print('2nd total reward:', total_reward_2)      
                
            if epoch < 10 or total_reward_2 == 0:
                # compute baseline
                baseline = reward.sum(dim=-1) / time_steps
                #baseline = Variable(baseline.data, volatile=True).cuda()
                baseline_rep = baseline.view(-1,1).repeat(1,N_samples)
                R_b = (reward - baseline_rep) # (R^t_i - b_t) with shape=(N*T)
            else:
                # compute baseline
                baseline = reward_2.sum(dim=-1) / time_steps
                #baseline = Variable(baseline.data, volatile=True).cuda()
                baseline_rep = baseline.view(-1,1).repeat(1,N_samples)
                R_b = (reward_2 - baseline_rep) # (R^t_i - b_t) with shape=(N*T)            
            '''

            
            
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
                        if index == 0:
                            param_vector = param.grad.data.clone().view(-1)
                        else:
                            param_vector = torch.cat((param_vector, param.grad.data.clone().view(-1)))
                        param.grad.data.zero_()
                    gradient_mu[t, l, :] = param_vector

            # compute gradient of policy w.r.t. mu
            difference = sample_location - mu_tensor.repeat(1, N_samples).view(1, time_steps, N_samples, 4)
            gradient_policy = difference / sigma[0]**2

            # compute gradient of W using chain rule
            # [0,t] means batch index 0 at time step t
            gradient_W = torch.bmm(gradient_policy.data.squeeze(), gradient_mu.cuda())
            factor1 = R_b.squeeze().view(time_steps,N_samples,1)
            factor2 = factor1.repeat(1,1,param_size)#.cuda()
            factor = factor2.view(time_steps, N_samples,param_size)
            gradient_G = torch.mul(gradient_W, factor.data)

            # sum up gradient of W for N_sample at all time_steps
            gradient_sum = gradient_G.sum(0).sum(0) / N_samples


            #### update parameters ####
            # update parameters of fc model using gradient ascent
            pointer = 0  # indicate the i-th parameter
            begin = 0  # indicate beginning position
            for param in rnn.parameters():
                param.data += learning_rate * gradient_sum[begin:begin + truncate_size[pointer]].cuda().view(size_list[pointer])
                begin += truncate_size[pointer]
                pointer += 1


# ### set the model in "testing mode" in order to close Dropout and save the model

# In[ ]:


rnn.eval()  


# ### save model and reward record

# In[ ]:


reward_record = np.array(reward_record)
dd.io.save('./saved_models/reward_300iters_04110960.h5')


# In[ ]:


torch.save(rnn.state_dict(),'./saved_models/DRLT_10classes_1000features_300iters_04110960.pth')

