#!/bin/python
# import matplotlib.pyplot as plt
import numpy as np
import os, sys, time
import numpy.random as rn

icon=['bx','ro','g*','c<']


class Sampler():
    """
    Sample data from a quadrant and label according to sample being within or outside unit circle.
    Normalize samples so labels are equally distributed. 
    """
    def __init__(self,
                alpha=1.0, 
                verbose=False,
                ntasks=2,
                dim=4,
                discriminator_offset=0.01, 
                distribution_offset=0.01, 
                uniform_width=1.25, 
                nsamples=50000,
                ntrain=4000,
                ntest=200):
        # seed
        rn.seed(512)
        stime=time.time()
        tasks={}
        
        # sample points for four tasks related to four quadrants
        for q in range(ntasks):
            #print("quadrant {0}".format(q))
            # step 1 sample points in quadrant uniformly from [0:1]
            if q == 0: # 1th quadrant
                xs=rn.uniform(distribution_offset,uniform_width,nsamples)
                ys=rn.uniform(distribution_offset,uniform_width,nsamples)
            elif q == 1: # 2nd quadrant
                xs=rn.uniform(-distribution_offset,-uniform_width,nsamples)
                ys=rn.uniform(distribution_offset,uniform_width,nsamples)
            elif q == 2: # 3th quadrant
                xs=rn.uniform(-distribution_offset,-uniform_width,nsamples)
                ys=rn.uniform(-distribution_offset,-uniform_width,nsamples)
            else: #4th quadrant
                xs=rn.uniform(distribution_offset,uniform_width,nsamples)
                ys=rn.uniform(-distribution_offset,-uniform_width,nsamples)
            samples=[]
            for i in range(len(xs)):
                sample=[xs[i],ys[i]]
                for j in range(dim-2): #add noise gausians in other dimensions
                    sample.append(rn.uniform(-uniform_width,uniform_width))
                samples.append(sample)
            tasks[q]=np.asarray(samples)
            if verbose: plt.plot(xs,ys,icon[q%ntasks])
        if verbose: plt.show()

        inputs={}
        labels={}

        # Step 3: Sample correct distribution given certain alpha
        for q in range(ntasks):
            pos_inputs=[]
            neg_inputs=[]
            other_quadrants=list(range(ntasks))
            del other_quadrants[q]
            while not (len(pos_inputs) == ntrain/2 and len(neg_inputs) == ntrain/2): 
                if rn.binomial(1,alpha): #1 if original task, 0 if data sampled from 1 of the others.
                    q_temp=q
                else: # pick from random other quadrant
                    q_temp=rn.choice(other_quadrants)
                sample=tasks[q_temp][0]
                # samples are popped from the distribution to avoid them to be reused.
                tasks[q_temp]=tasks[q_temp][1:]
                if np.sqrt(np.sum(sample**2))>1+discriminator_offset and len(neg_inputs) < ntrain/2:
                    neg_inputs.append(sample) 
                #elif np.sqrt(np.sum(sample**2))<1:
                elif np.sqrt(np.sum(sample**2))<1-discriminator_offset and len(pos_inputs) < ntrain/2:
                    pos_inputs.append(sample) 
                else: #discard samples on border as task becomes too hard 
                    pass
            pos_inputs=pos_inputs[:int(ntrain/2)]
            neg_inputs=neg_inputs[:int(ntrain/2)]
            inputs[q]=pos_inputs+neg_inputs
            rn.shuffle(inputs[q])
            labels[q]=[]
            for sample in inputs[q]:
                if np.sqrt(np.sum(sample**2))>1+discriminator_offset:
                    labels[q].append(0) 
                elif np.sqrt(np.sum(sample**2))<1-discriminator_offset:
                    labels[q].append(1) 
                else: #discard samples on border as task becomes too hard 
                    pass
            #if verbose: plt.plot(sample[0],sample[1],icon[q])
            print('{0}:proportion positive/total={1}'.format(q,float(sum(labels[q]))/len(labels[q])))
        
        # draw some extra test data to evaluate
        # label points according to distance to center
        test_inputs={}
        test_labels={}
        for q in range(ntasks):
            test_inputs_pos=[]
            test_inputs_neg=[]
            while len(test_inputs_pos)<ntest/2 or len(test_inputs_neg)<ntest/2:
                sample=tasks[q][0]
                tasks[q]=tasks[q][1:]
                # map sample to 2D plane for visualizations
                if len(sample)>2: sample[2:]=0
                if np.sqrt(np.sum(sample**2)) > 1+discriminator_offset:
                    test_inputs_neg.append(sample)
                    #test_inputs[q].append(sample)
                    #test_labels[q].append(0)
                #elif np.sqrt(np.sum(sample**2)) < 1:
                elif np.sqrt(np.sum(sample**2)) < 1-discriminator_offset:
                    test_inputs_pos.append(sample)
                    #test_inputs[q].append(sample)
                    #test_labels[q].append(1)
                else:
                    pass
            test_inputs[q]=np.asarray(test_inputs_pos[:int(ntest/2)]+test_inputs_neg[:int(ntest/2)])
            test_labels[q]=[1]*int(ntest/2)+[0]*int(ntest/2)
            #print test_inputs[q].shape
        # Check distribution
        for q in range(ntasks):
            positive_points=[test_inputs[q][i] for i in range(len(test_inputs[q])) if test_labels[q][i]==1]
            negative_points=[test_inputs[q][i] for i in range(len(test_inputs[q])) if test_labels[q][i]==0]
        
            if verbose: plt.scatter([p[0] for p in positive_points],[p[1] for p in positive_points],color='blue')
            if verbose: plt.scatter([p[0] for p in negative_points],[p[1] for p in negative_points],color='red')
        if verbose: plt.show()
        print("sampling data duration: ",time.time()-stime)

        self.tasks=tasks
        self.inputs=inputs
        self.labels=labels
        self.test_inputs=test_inputs
        self.test_labels=test_labels