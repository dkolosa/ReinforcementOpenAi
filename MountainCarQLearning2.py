#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 23:39:13 2017

@author: sezan92
"""
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  2 23:39:13 2017

@author: sezan92
"""
# In[1] Libraries
import gym
import numpy as np
from QLearn import QL
# In[2]
env = gym.make('MountainCar-v0')
s = env.reset()


# In[Constants]
legal_actions=env.action_space.n
actions = [0,1,2]
gamma =0.99
lr =0.5
num_episodes =30000
epsilon =0.5
epsilon_decay =0.99

# In[Discretize Bins]
N_BINS = [10,10]

MIN_VALUES = [0.6,0.07]
MAX_VALUES = [-1.2,-.07]
BINS = [np.linspace(MIN_VALUES[i], MAX_VALUES[i], N_BINS[i]) for i in range(2)]
rList =[]
def discretize(obs):
       return tuple([int(np.digitize(obs[i], BINS[i])) for i in range(2)])

# In[Class Q]
Q = {}
policy ={}
legal_actions =3
actions =[0,1,2]
gamma = 0.99
lr =0.5
QL = QL(Q,policy,legal_actions,actions,gamma,lr)
# In[4]
for i in range(num_episodes):
    s_raw= env.reset()
    s = discretize(s_raw)
    rAll =0
    d = False
    j = 0
    for j in range(200):
        
        #epsilon greedy. to choose random actions initially when Q is all zeros
        if np.random.random()< epsilon:
            a = np.random.randint(0,legal_actions)
            epsilon = epsilon*epsilon_decay
        else:
            a =QL.action(s)
        s1_raw,r,d,_ = env.step(a)
        rAll=rAll+r
        s1 = discretize(s1_raw)
        env.render()
        if d:
            if rAll<-199:
                r =-100
                QL.learn(s,a,s1,r,d)
                print("Failed! Reward %d"%rAll)
            elif rAll>-199:
                print("Passed! Reward %d"%rAll)
            break
        QL.learn(s,a,s1,r,d)
        if j==199:
            print("Reward %d after full episode"%(rAll))
            
        s = s1
env.close()  
