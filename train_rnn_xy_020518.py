
# coding: utf-8

# In[3]:


import os
import time
from RNN import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image


# ### time begin

# In[4]:


start_time = time.time()


# ### use gpu

# In[5]:


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
use_gpu = torch.cuda.is_available()


# ### speed up and save GPU memory

# In[6]:


torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


# ### load dataset, namely feature+location a.k.a. combo

# In[7]:


combo_all = dd.io.load('yolo_combo1002_20classes_S16_lambdt25.h5') # utilize pad ground truth


# In[8]:


combo_all[0].size()


# In[9]:


# append (x,y,w,h) instead of (x,y,sqrt(w),sqrt(h))
#combo_all[0][-2:,:] = combo_all[0][-2:,:]**2


# In[10]:


gt_all = dd.io.load('../20classes_normal_bboxes_all_sqrtwh_list.h5')


# In[11]:


gt_all_xy = []


# In[12]:


for i in range(len(gt_all)):
    n_data = gt_all[i].shape[0]
    gt_xy = np.zeros((n_data, 2))
    for j in range(n_data):
        gt_xy[j] = gt_all[i][j][:2]
    gt_all_xy.append(gt_xy)


# In[13]:


len(gt_all_xy)


# In[14]:


gt_all_xy[0].shape


# In[15]:


combo_all[0].size()


# In[16]:


gt_all_xy[0][0]


# ### set hyperparameters

# In[41]:


#n_data = 50                   # 20 data needs two process
n_features = combo_all[0].size(1)              # YOLO output plus padding ground truth location or zeros
hidden_size = input_size = 1002 # input size is the size of input of RNN
time_steps = 10               # a.k.a. N=10 in the paper
n_batch = 1                   # number of batch
n_layers = 1                  # number of recurrent layers
N_samples = 3                 # number of sample locations for each time step
n_epochs = 40                 # number of epochs for training
sigma = torch.FloatTensor(2).cuda()  # variance for (x,y)
sigma[:] = 0.01#0.05
learning_rate = 0.0006 #0.0006
n_dataset = 1 # only for bird
weight_avg = 1
weight_max = 10
train_part = 3


# In[33]:


#learning_rate *= ratio_wh


# In[14]:


#learning_rate = learning_rate.max(1)


# In[15]:


#learning_rate = torch.from_numpy(learning_rate)


# ### initialize FC module and RNN module

# In[18]:


rnn = RNN_module(n_features, input_size, hidden_size, n_batch).cuda() # run the module on GPU


# ### initialize FC layer

# In[19]:


for m in rnn.modules():
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


# In[80]:


# load pre-trained model
rnn = RNN_module(n_features, input_size, hidden_size, n_batch)
rnn.load_state_dict(torch.load('./newest/tracker_ants1_130iters_10steps_S16.pth'))
rnn.eval()
if use_gpu:
    rnn.cuda()


# ### set the model in "training mode"

# In[42]:


rnn.train()


# ### function in order to plot the image data

# In[21]:


# load class files
class_file = '../../routine_generate_vot2017_train/vot2017/list_20classes.txt'
CLASSES = [line.rstrip('\n') for line in open(class_file)]


# In[22]:


CLASSES


# In[23]:


root_folder = '../../routine_generate_vot2017_train/vot2017/vot2017/'
vot_folder = os.path.join(root_folder, CLASSES[0])


# In[87]:


def plot_img(folder, sampled_img_index, bboxes, gt):
    #print(folder)
    #print(img_name)
    #print(bboxes)
    print(len(bboxes))
    print(bboxes[0].shape)
    print('hello')
    fig,ax = plt.subplots(2,5, figsize=(30, 30))
    #for k, index in enumerate(sampled_img_index):
    
    index = 0
    for i in range(len(bboxes)):
        for j in range(len(bboxes[0])):
            img_name = '{0:08}'.format(index+1) + '.jpg' # ith image in subfolder
            img = np.array(Image.open(os.path.join(folder, img_name)))
            width, height, _ = img.shape
            train_location = bboxes[i][j] * np.array([width, height])
            gt_one = gt[index] * np.array([width, height, np.sqrt(width), np.sqrt(height)])
            #print(train_location)
            train_location[2:] = train_location[2:]**2
            gt_one[2:] = gt_one[2:]**2
            w = 0.02 * width
            h = 0.02 * height
            x = train_location[0] - w / 2
            y = train_location[1] + h / 2
            w_gt = 0.02 * width #gt_one[2]
            h_gt = 0.02 * height #gt_one[3]
            x_gt = gt_one[0] - w_gt / 2
            y_gt = gt_one[1] - h_gt / 2
            print("****")
            print(x_gt,y_gt,w_gt,h_gt)
            print(x,y,w,h)
            print("****")
            # Create figure and axes
            #print(x,y,w,h)
            #print(x_gt,y_gt,w_gt,h_gt)
            #print("******")

            # Display the image
            ax[index].imshow(img)
            # Create a Rectangle patch
            rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='b',facecolor='none')
            rect_gt = patches.Rectangle((x_gt,y_gt),w_gt,h_gt,linewidth=1,edgecolor='r',facecolor='none')
            # Add the patch to the Axes
            ax[index].add_patch(rect)
            ax[index].add_patch(rect_gt)
            index += 1
        plt.show()
        fig.savefig('exaple.png', bbox_inches='tight')


# In[105]:


index = 0
row_index = 0
col_index = 0
gt = gt_all[0]
fig,ax = plt.subplots(20,5, figsize=(100,300))
#fig,ax = plt.subplots(20,5)
#fig.tight_layout()
for i in range(len(bboxes_list)):
    print(i)
    for j in range(len(bboxes_list[0])):
        img_name = '{0:08}'.format(index+1) + '.jpg' # ith image in subfolder
        img = np.array(Image.open(os.path.join(vot_folder, img_name)))
        width, height, _ = img.shape
        train_location = bboxes_list[i][j] * np.array([width, height])
        gt_one = gt[index] * np.array([width, height, np.sqrt(width), np.sqrt(height)])
        #print(train_location)
        train_location[2:] = train_location[2:]**2
        gt_one[2:] = gt_one[2:]**2
        w = 0.02 * width
        h = 0.02 * height
        x = train_location[0] - w / 2
        y = train_location[1] + h / 2
        w_gt = 0.02 * width #gt_one[2]
        h_gt = 0.02 * height #gt_one[3]
        x_gt = gt_one[0] - w_gt / 2
        y_gt = gt_one[1] - h_gt / 2
        #print("****")
        #print(x_gt,y_gt,w_gt,h_gt)
        #print(x,y,w,h)
        #print("****")
        # Create figure and axes
        #print(x,y,w,h)
        #print(x_gt,y_gt,w_gt,h_gt)
        #print("******")

        # Display the image
        ax[row_index,col_index].imshow(img)
        # Create a Rectangle patch
        rect = patches.Rectangle((x,y),w,h,linewidth=1,edgecolor='b',facecolor='none')
        rect_gt = patches.Rectangle((x_gt,y_gt),w_gt,h_gt,linewidth=1,edgecolor='r',facecolor='none')
        # Add the patch to the Axes
        ax[row_index,col_index].add_patch(rect)
        ax[row_index,col_index].add_patch(rect_gt)
        ax[row_index,col_index].set_yticklabels([])
        ax[row_index,col_index].set_xticklabels([])
        index += 1
        
        col_index += 1
        if col_index > 4:
            col_index = 0
            row_index += 1
plt.show()


# In[106]:


fig.savefig('exaple.png', bbox_inches='tight')


# In[77]:


ax.shape


# ### training process

# In[47]:


def warm_start(combo_train, n_layers, n_batch, hidden_size):
    h_state = Variable(torch.randn(n_layers, n_batch, hidden_size)).cuda()
    c_state = Variable(torch.randn(n_layers, n_batch, hidden_size)).cuda()
    input = combo_train[0:0 + 1]
    input = Variable(input).cuda()
    #for t in range(5):  # times for the warm start
    #h_state, c_state = rnn(input[0,:].view(1,-1), h_state, c_state)  


# In[59]:


4%5


# In[71]:


reward_record = []
reward_2nd_record = []
for epoch in range(n_epochs): # run the policy for n_epochs
    
    # pick a few sampled imgs and| plot it in order to check the whole process
    base_index = 5
    base_number = np.array([0, 1, 2, 3])
    sampled_img_index = base_number * 10 + base_index
    sampled_img = []
    bboxes_list = []
    #sigma[:] = -0.00153 * epoch + 0.06 # linear decay for sigma
    #if img_index is in list then plot it

    #print(sampled_img)
    #print('epoch %d' %(epoch+1))
    for ith_dataset in range(n_dataset):
        subfolder_name = CLASSES[ith_dataset]
        img_folder = os.path.join('../routine_generate_vot2017_train/vot2017', subfolder_name)
        #ith_dataset = 0 # only for bolt
        n_subtrain = gt_all_xy[ith_dataset].shape[0] // train_part // time_steps
        n_subtrain *= time_steps
        combo_train = combo_all[ith_dataset][:n_subtrain]
        gt_train = gt_all_xy[ith_dataset][:n_subtrain]
        #gt = Variable(torch.from_numpy(gt_train).float(), volatile=True) # run this variable on GPU
        #combo = Variable(combo_train).cuda() # run this variable on GPU
        
        loop_begin = 0  # loop pointer
        gt_pointer = 0  # ground truth array pointer
        #print('')
        print('epoch %d, the dataset %d' %(epoch+1, ith_dataset+1))
        # randomly generate initial hidden and cell states at begining of each epoch
        h_state = Variable(torch.randn(n_layers, n_batch, hidden_size)).cuda()
        c_state = Variable(torch.randn(n_layers, n_batch, hidden_size)).cuda()
        
        '''
        # with warm start for initialization
        input_warm = combo_train[0:0 + time_steps]
        input_warm = Variable(input_warm).cuda()
        for t in range(10):  # warm start for 5 steps
            # for each time step apply RNN and get h_state, c_state for current time step
            #one_input = input[t,:].view(1,-1)
            #one_input = Variable(one_input).cuda()
            h_state, c_state = rnn(input_warm[t,:].view(1,-1), h_state, c_state)        
        '''

        
        for loop in range(n_subtrain // time_steps): # for each 10 frames
            
            # set learning rate
            if loop <= 20:
                #learning_rate = 0.0006
                learning_rate = 0.00001
            else:
                learning_rate = 0.00006
            
            #### forward pass ####
            # get input of RNN from FC layer output
            input = combo_train[loop_begin:loop_begin + time_steps]
            input = Variable(input).cuda()
            #print(input.size())

            # compute mu from hidden states and sample N_samples location for each time step t
            #sample_location = Variable(torch.FloatTensor(n_batch, time_steps, N_samples, 4), volatile=True).cuda()
            # for x any y 
            sample_location = Variable(torch.FloatTensor(n_batch, time_steps, N_samples, 2), volatile=True).cuda()
            # 2 is location (x,y)
            mu_tensor = Variable(torch.FloatTensor(time_steps, 2)).cuda() # for x and y
            for index_batch in range(n_batch):  # index for batch
                for t in range(time_steps):  # index for time_step
                    # for each time step apply RNN and get h_state, c_state for current time step
                    #one_input = input[t,:].view(1,-1)
                    #one_input = Variable(one_input).cuda()
                    h_state, c_state = rnn(input[t,:].view(1,-1), h_state, c_state)
                    # compute network output mean mu of location which contains (x,y,w,h)
                    mu = h_state[0, 0][-2:]  # for x and y prediction
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
            
            rep_gt = gt.repeat(1, N_samples).view(1, time_steps, N_samples, 2) # for x and y prediction
            gt_pointer += time_steps
            abs_sub = (sample_location - rep_gt).abs()
        
            # visualize the training images and save it 
            bbox_mean = sample_location.mean(dim=-2) # in order to plot bounding box on img
            #bbox_mean = bbox_mean[0,base_index,:].cpu().data.numpy() # numpy array format
            bbox_mean = bbox_mean[0].cpu().data.numpy()
            bboxes_list.append(bbox_mean)
                
            
            avg_val = abs_sub.mean(dim=-1)
            max_val = abs_sub.max(dim=-1)[0]
            reward = -weight_avg * avg_val - weight_max * max_val
            G_display = reward.sum().cpu().data.numpy()[0]
            reward_record.append(G_display)
            print('1st total reward:', G_display)            
            # compute baseline
            baseline = reward.sum(dim=-1) / time_steps
            #baseline = Variable(baseline.data, volatile=True).cuda()
            baseline_rep = baseline.view(-1,1).repeat(1,N_samples)
            R_b = (reward - baseline_rep) # (R^t_i - b_t) with shape=(N*T)                               
            
            
            # to the next 10 time steps
            loop_begin += time_steps
            
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
            gradient_mu = torch.FloatTensor(time_steps, 2, param_size)  ## for x and y only
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
            difference = sample_location - mu_tensor.repeat(1, N_samples).view(1, time_steps, N_samples, 2) # for x and y
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
            del factor1,factor2,factor,gradient_mu,gradient_policy,gradient_sum,gradient_G,
            
    # plot the sampled images
    plot_img(vot_folder, sampled_img_index, bboxes_list, gt_all[0]) # for ant


# In[ ]:


bboxes_list


# In[ ]:


print("\n--- it costs %.4s minutes ---" % ((time.time() - start_time)/60))


# ### set the model in "testing mode" in order to close Dropout and save the model

# In[39]:


rnn.eval()  


# ### save model and reward record

# In[40]:


#torch.save(rnn.state_dict(),'./tracker_ants1_xy_S16.pth')
print('model has saved!')


# In[ ]:


reward_record = np.array(reward_record)
#dd.io.save('./newest/reward_3classes_test_5steps_S16.h5', reward_record)
print('reward has saved!')

