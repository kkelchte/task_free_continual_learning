#!/usr/bin/python

# import matplotlib.pyplot as plt
import numpy as np
import os, sys, time
import numpy.random as rn
import torch

class Task_free_continual_learning():

    def __init__(self,
                verbose=False,
                seed=123,
                dev='cpu',
                dim=4,
                hidden_units=100,
                learning_rate=0.005,
                ntasks=2,
                gradient_steps=5,
                loss_window_length=5,
                loss_window_mean_threshold=0.2,
                loss_window_variance_threshold=0.1, 
                MAS_weight=0.5,
                recent_buffer_size=30,
                hard_buffer_size=30):

        torch.manual_seed(seed)
        device = torch.device(dev)
        
        # Save settings
        self.verbose=verbose
        self.dim=dim
        self.ntasks=ntasks
        self.gradient_steps=gradient_steps
        self.loss_window_length=loss_window_length
        self.loss_window_mean_threshold=loss_window_mean_threshold
        self.loss_window_variance_threshold=loss_window_variance_threshold
        self.MAS_weight=MAS_weight
        self.recent_buffer_size=recent_buffer_size
        self.hard_buffer_size=hard_buffer_size



        # Create pytorch network
        self.model = torch.nn.Sequential(
                  torch.nn.Linear(dim, hidden_units, bias=True),
                  torch.nn.ReLU(),
                  torch.nn.Linear(hidden_units, hidden_units, bias=True),
                  torch.nn.ReLU(),
                  torch.nn.Linear(hidden_units, 2, bias=False),
        ).to(device)
        # define loss and optimizer
        self.loss_fn = torch.nn.MSELoss(reduction='none')
        self.optimizer=torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        
        # initialize model
        for p in self.model.parameters():
            torch.nn.init.normal_(p, 0, 0.1)


    def method(self,
                data, 
                use_hard_buffer=False,
                continual_learning=False):
        count_updates=0
    
        stime=time.time()
        losses=[]
        test_loss={i:[] for i in range(self.ntasks)}
        recent_buffer=[]
        hard_buffer=[]
        # loss dectection
        loss_window=[]
        loss_window_means=[]
        loss_window_variances=[]
        update_tags=[]
        new_peak_detected=True
        # MAS regularization: list of 3 weights vectors as there are 3 layers.
        star_variables=[]
        omegas=[] #initialize with 0 importance weights
        for t in range(self.ntasks):
            for s in range(len(data.inputs[t])):
                # plt.scatter(inputs[t][s][0],inputs[t][s][1],color='red' if labels[t][s]==0 else 'blue')
                # save experience in replaybuffer
                recent_buffer.append({'state':data.inputs[t][s],
                                     'trgt':data.labels[t][s]})
                if len(recent_buffer) > self.recent_buffer_size:
                    del recent_buffer[0]
                
                # Train model on replaybuffer when it is full:
                if len(recent_buffer) == self.recent_buffer_size:
                    msg='task: {0} step: {1}'.format(t,s)

                    # get frames new frames from recent buffer
                    x=[_['state'] for _ in recent_buffer]
                    y=[_['trgt'] for _ in recent_buffer]
                    
                    if use_hard_buffer and len(hard_buffer) != 0:
                        xh=[_['state'] for _ in hard_buffer]
                        yh=[_['trgt'] for _ in hard_buffer]

                    # train
                    for gs in range(self.gradient_steps):
                        # evaluate recent buffer
                        y_pred = self.model(torch.from_numpy(np.asarray(x).reshape(-1,self.dim)).type(torch.float32))
                        y_sup=torch.zeros(len(y),2).scatter_(1,torch.from_numpy(np.asarray(y).reshape(-1,1)).type(torch.LongTensor),1.).type(torch.FloatTensor)
                        recent_loss = self.loss_fn(y_pred,y_sup)
                        total_loss = torch.sum(self.loss_fn(y_pred,y_sup))
                    
                        if use_hard_buffer and len(hard_buffer) != 0:
                            # evaluate hard buffer
                            yh_pred = self.model(torch.from_numpy(np.asarray(xh).reshape(-1,self.dim)).type(torch.float32))
                            yh_sup=torch.zeros(len(yh),2).scatter_(1,torch.from_numpy(np.asarray(yh).reshape(-1,1)).type(torch.LongTensor),1.).type(torch.FloatTensor)
                            
                            hard_loss = self.loss_fn(yh_pred,yh_sup)
                            total_loss += torch.sum(self.loss_fn(yh_pred,yh_sup))
                        
                        # keep train loss for loss window
                        if gs==0: first_train_loss=total_loss.detach().numpy()
                        
                        # add MAS regularization to train loss...
                        if continual_learning and len(star_variables)!=0 and len(omegas)!=0:
                            for pindex, p in enumerate(self.model.parameters()):
                                total_loss+=self.MAS_weight/2.*torch.sum(torch.from_numpy(omegas[pindex]).type(torch.float32)*(p-star_variables[pindex])**2)
                                                   
                        # train self.model
                        self.optimizer.zero_grad()
                        torch.sum(total_loss).backward()
                        self.optimizer.step()
                
                    # save training accuracy on total batch
                    if use_hard_buffer and len(hard_buffer) != 0:
                        xt=x+xh
                        yt=y+yh
                    else:
                        xt=x[:]
                        yt=y[:]
                    yt_pred = self.model(torch.from_numpy(np.asarray(xt).reshape(-1,self.dim)).type(torch.float32))
                    accuracy = np.mean(np.argmax(yt_pred.detach().numpy(),axis=1)==yt)
                    msg+=' recent loss: {0:0.3f}'.format(np.mean(recent_loss.detach().numpy()))
                    if use_hard_buffer and len(hard_buffer) != 0:
                        msg+=' hard loss: {0:0.3f}'.format(np.mean(hard_loss.detach().numpy()))
                    losses.append(np.mean(accuracy))
                    
                    
                    # add loss to loss_window and detect importance weight update
                    loss_window.append(np.mean(first_train_loss))
                    if len(loss_window)>self.loss_window_length: del loss_window[0]
                    loss_window_mean=np.mean(loss_window)
                    loss_window_variance=np.var(loss_window)
                    if not new_peak_detected and loss_window_mean > last_loss_window_mean+np.sqrt(last_loss_window_variance):
                        new_peak_detected=True                    
                    if continual_learning and loss_window_mean < self.loss_window_mean_threshold and loss_window_variance < self.loss_window_variance_threshold and new_peak_detected:
                        count_updates+=1
                        update_tags.append(0.01)
                        last_loss_window_mean=loss_window_mean
                        last_loss_window_variance=loss_window_variance
                        new_peak_detected=False
                        
                        # calculate importance weights and update star_variables
                        gradients=[0 for p in self.model.parameters()]
                        
                        # calculate imporatance based on each sample in recent + hardbuffer
                        for sx in [_['state'] for _ in hard_buffer]:
                            self.model.zero_grad()
                            y_pred=self.model(torch.from_numpy(np.asarray(sx).reshape(-1,self.dim)).type(torch.float32))
                            torch.norm(y_pred, 2, dim=1).backward()
                            for pindex, p in enumerate(self.model.parameters()):
                                g=p.grad.data.clone().detach().numpy()
                                gradients[pindex]+=np.abs(g)
                                
                        omegas_old = omegas[:]
                        omegas=[]
                        star_variables=[]
                        for pindex, p in enumerate(self.model.parameters()):
                            if len(omegas_old) != 0:
                                omegas.append(1/count_updates*gradients[pindex]+(1-1/count_updates)*omegas_old[pindex])
                            else:
                                omegas.append(gradients[pindex])
                            star_variables.append(p.data.clone().detach())
                        
                    else:
                        update_tags.append(0)
                    loss_window_means.append(loss_window_mean)
                    loss_window_variances.append(loss_window_variance)

                    #update hard_buffer
                    if use_hard_buffer:                    
                        if len(hard_buffer) == 0:
                            loss=recent_loss.detach().numpy()
                        else:
                            loss=torch.cat((recent_loss, hard_loss))
                            loss=loss.detach().numpy()
                            
                        hard_buffer=[]
                        loss=np.mean(loss,axis=1)
                        sorted_inputs=[np.asarray(lx) for _,lx in reversed(sorted(zip(loss.tolist(),xt),key= lambda f:f[0]))]
                        sorted_targets=[ly for _,ly in reversed(sorted(zip(loss.tolist(),yt),key= lambda f:f[0]))]
                            
                        for i in range(min(self.hard_buffer_size,len(sorted_inputs))):
                            hard_buffer.append({'state':sorted_inputs[i],
                                               'trgt':sorted_targets[i]})
                    #evaluate on test set
                    for i in range(self.ntasks):
                        y_pred=self.model(torch.from_numpy(data.test_inputs[i].reshape(-1,self.dim)).type(torch.float32))
                        y_sup=torch.zeros(len(data.test_inputs[i]),2).scatter_(1,torch.from_numpy(np.asarray(data.test_labels[i]).reshape(-1,1)),1.).type(torch.FloatTensor)
                        #loss=loss_fn(y_pred,y_sup).detach().numpy()
                        test_accuracy=np.mean(np.argmax(y_pred.detach().numpy(),axis=1)==data.test_labels[i])
                        test_loss[i].append(test_accuracy)
                        msg+=' test[{0}]: {1:0.3f}'.format(i,test_accuracy)
                    if self.verbose:
                        print(msg)
                    # empty recent buffer after training couple of times
                    recent_buffer = []
            
        if False and use_hard_buffer:
            xs_pos=[_['state'][0] for _ in hard_buffer if _['trgt']==1]
            ys_pos=[_['state'][1] for _ in hard_buffer if _['trgt']==1]
            xs_neg=[_['state'][0] for _ in hard_buffer if _['trgt']==0]
            ys_neg=[_['state'][1] for _ in hard_buffer if _['trgt']==0]
            plt.scatter(xs_pos,ys_pos,color='blue')
            plt.scatter(xs_neg,ys_neg,color='red')
            plt.title('replay buffer')
            plt.show()
            
        if False:
            for q in range(self.ntasks):
                y_pred=model(torch.from_numpy(test_inputs[q].reshape(-1,self.dim)).type(torch.float32)).detach().numpy()
                positive_points=[test_inputs[q][i] for i in range(len(test_inputs[q])) if np.argmax(y_pred[i])==1]
                negative_points=[test_inputs[q][i] for i in range(len(test_inputs[q])) if np.argmax(y_pred[i])==0]
                plt.scatter([p[0] for p in positive_points],[p[1] for p in positive_points],color='blue')
                plt.scatter([p[0] for p in negative_points],[p[1] for p in negative_points],color='red')
            plt.axis('off')
            plt.show()
        
        print("duration: {0}minutes, count updates: {1}".format((time.time()-stime)/60., count_updates))

        return losses, loss_window_means, update_tags, loss_window_variances, test_loss