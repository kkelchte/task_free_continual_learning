#!/usr/bin/python

# import matplotlib.pyplot as plt
import numpy as np
import os, sys, time
import matplotlib.patches as mpatches

from method import Task_free_continual_learning
from sampler import Sampler

ntasks=2

def experiment(data, learning_object, tags=['Online Continual','Online Continual No Hardbuffer','Online','Online No Hardbuffer']):
    training_losses={}
    test_losses={}
    loss_window_means={}
    update_tags={}
    loss_window_variances={}
    settings={'Online Continual':(True, True),
             'Online Continual No Hardbuffer':(False, True),
             'Online':(True, False),
             'Online No Hardbuffer':(False, False)}
    colors={'Online Continual':'C2',
             'Online Continual No Hardbuffer':'C3',
             'Online':'C4',
             'Online No Hardbuffer':'C5'}

    for tag in tags:
        print("\n{0}".format(tag))
        # losses, loss_window_means, update_tags, loss_window_variances, test_loss=experiment(use_hard_buffer,continual_learning)
        results=learning_object.method(data,
                                        use_hard_buffer=settings[tag][0],
                                        continual_learning=settings[tag][1])
        training_losses[tag], loss_window_means[tag], update_tags[tag], loss_window_variances[tag], test_losses[tag] = results
    # Plot loss window mean, variance and update for each tag
    if False and 'Online Continual' in tags or 'Online Continual No Hardbuffer' in tags:
        for dataname in ['loss_window_means','update_tags','loss_window_variances']:
            legend=[]
            plt.title(dataname)
            #for i in range(ntasks): plt.axvline(x=(i+1)*ntrain,color='gray')
            for tag in tags:    
                plt.plot(eval(dataname)[tag],color=colors[tag])
                legend.append(mpatches.Patch(color=colors[tag], label=tag))
            plt.legend(handles=legend)
            plt.axis('off')
            plt.show()
    #print loss_window_means
    
    # Plot training loss
    if False:
        legend=[]
        plt.title('training accuracy')
        #for i in range(ntasks): plt.axvline(x=(i+1)*ntrain,color='gray')
        for tag in tags:    
            plt.plot(training_losses[tag][::10],color=colors[tag])
            legend.append(mpatches.Patch(color=colors[tag], label=tag))
        plt.legend(handles=legend)
        plt.axis('off')
        plt.show()
        
    # Plot test loss
    if False:
        subsample=1
        for task in range(2):
            legend=[]
            plt.title('test loss task {0}'.format(task))
            plt.ylim((0,1))
            #for i in range(ntasks): plt.axvline(x=(i+1)*ntrain,color='gray')
            for tag in tags:  
                plt.plot(np.arange(0,len(test_losses[tag][task]),subsample),test_losses[tag][task][::subsample],color=colors[tag])
                legend.append(mpatches.Patch(color=colors[tag], label=tag))
            plt.legend(handles=legend)
            plt.axis('off')
            plt.show()
            print(tag,task,test_losses[tag][task][-1]*100)

    # Get final average accuracy for each tag:
    for tag in tags:
        print("{0}: {1}".format(tag,np.mean([test_losses[tag][task][-1]*100 for task in range(ntasks)])))
        for task in range(ntasks): print("{0}: task {1}: {2}".format(tag,task,test_losses[tag][task][-1]*100))
    return [np.mean([test_losses[tag][task][-1]*100 for task in range(ntasks)]) for tag in sorted(tags)]


def main():

    # number or tasks or quadrants
    ntasks=2
    dim=4
    data = Sampler(alpha=1.0,
                    verbose=False,
                    ntasks=ntasks,
                    dim=dim,
                    discriminator_offset=0.05, 
                    distribution_offset=0.5, 
                    uniform_width=1., 
                    nsamples=1000000,
                    ntrain=10000,
                    ntest=200)

    # learning_object=Task_free_continual_learning(verbose=False,
    #                                                     seed=123,
    #                                                     dev='cpu',
    #                                                     dim=dim,
    #                                                     hidden_units=100,
    #                                                     learning_rate=0.005,
    #                                                     ntasks=ntasks,
    #                                                     gradient_steps=5,
    #                                                     loss_window_length=5,
    #                                                     loss_window_mean_threshold=0.2,
    #                                                     loss_window_variance_threshold=0.1,                                                         
    #                                                     MAS_weight=0.5,
    #                                                     recent_buffer_size=20,
    #                                                     hard_buffer_size=5)


    learning_object=Task_free_continual_learning(verbose=False,
                                                        seed=123,
                                                        dev='cpu',
                                                        dim=dim,
                                                        hidden_units=100,
                                                        learning_rate=0.005,
                                                        ntasks=ntasks,
                                                        gradient_steps=5,
                                                        loss_window_length=5,
                                                        loss_window_mean_threshold=0.2,
                                                        loss_window_variance_threshold=0.1,                                                         
                                                        MAS_weight=0.5,
                                                        recent_buffer_size=20,
                                                        hard_buffer_size=5)

    tags=['Online No Hardbuffer', 'Online Continual']
    experiment(data, learning_object, tags)




if __name__ == '__main__':
    main()    
