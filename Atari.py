#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 20:19:23 2017

@author: sezan92
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 17:54:48 2017

@author: sezan92
"""

# In[1] importing libraries

import gym 
import tensorflow as tf
import numpy as np
import cv2
# In[2]

env = gym.make('AirRaid-v0')
env.reset()

# In[3] Class for Tensorflow
class QNetwork:
    def __init__(self,learning_rate=0.01,action_size =6,
                 name = 'QNetwork'):
        with tf.variable_scope(name):
            self.inputs_ = tf.placeholder(tf.float32,
                                          [None,84,84,4])
            #self.inputs_ = tf.reshape(self.inputs_NotImage,[-1,84,84,4])
            self.actions_= tf.placeholder(tf.int32,[None],
                                         name = 'actions')
            one_hot_actions = tf.one_hot(self.actions_,action_size)
            self.targetQs_ = tf.placeholder(tf.float32,[None],name=
                                            'target')
            # 84 x 84 x 4
            self.conv1 = tf.layers.conv2d(self.inputs_,32,(8,8),
                                          strides = 4,
                                          activation=tf.nn.relu)
            # 20 x 20 x32
            self.conv2= tf.layers.conv2d(self.conv1,64,(4,4),
                                         strides=2,activation=tf.nn.relu)
            # 9 x 9 x 64
            self.conv3 = tf.layers.conv2d(self.conv2,64,(3,3),
                                          strides=1,
                                          activation=tf.nn.relu)
            #7 x 7 x 64
            self.flat = tf.contrib.layers.flatten(self.conv3)
            self.fc1=tf.contrib.layers.fully_connected(self.flat,
                                                       512,activation_fn=tf.nn.relu)
            #512
            self.output =tf.contrib.layers.fully_connected(self.fc1,
                                                        action_size,
                                                        activation_fn=None)
            #18
            self.Q = tf.reduce_sum(tf.multiply(self.output,one_hot_actions),
                                   axis =1)
            self.loss = tf.reduce_mean(tf.square(self.targetQs_-self.Q))
            self.opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

# In[4] Class for Experience Replay

from collections import deque

class Memory():
    def __init__(self,max_size =1000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self,experience):
        self.buffer.append(experience)
    def sample(self,batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)),
                              size=batch_size,
                              replace = False)
        return [self.buffer[ii] for ii in idx]
def StateProcess(env,action):
    rewards=0
    stateList = []     
    for ii in range(4):
        #if ii>0:
            #action=0
        state1,reward,done,info = env.step(action)
        stateGray= cv2.cvtColor(state1,cv2.COLOR_RGB2GRAY)
        stateGray = cv2.resize(stateGray,(84,84))
        stateList.append(stateGray)
        rewards+=reward
    state = np.dstack(stateList)
    return state,rewards,done,info
    
# In[5] Hyperparameters

train_episodes = 1000
max_steps = 2000000
gamma = 0.8

explore_start = 1.0
explore_stop = 0.01
decay_rate = 0.0001

#Network parameters
hidden_size = 64
learning_rate=0.01

#Memory parameters
memory_size = 10000
batch_size = 20
pretrain_length = batch_size*100
tf.reset_default_graph()
mainQN = QNetwork(name = 'main',
                  learning_rate = learning_rate)

# In[6] Populate the experience
action = env.action_space.sample()
env.reset()
state,reward,done,_ = StateProcess(env,action)
memory = Memory(max_size=memory_size)
for ii in range(pretrain_length):
    env.render()
    action = env.action_space.sample()
    next_state,reward,done,_ = StateProcess(env,action)
    if done:
        next_state = np.zeros(state.shape)
        memory.add((state,action,reward,next_state))
        env.reset()
        action = env.action_space.sample()
        state,reward,done,_ = StateProcess(env,action)
    else:
        memory.add((state,action,reward,next_state))
        state=next_state
# In[7] Tensorflow training
saver =tf.train.Saver()
reward_list =[]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step=0
    for ep in range(1,train_episodes):
        total_reward=0
        t=0
        while t==0:
            step+=1
            env.render()
            explore_p =explore_stop+(explore_start-explore_stop)*np.exp(-decay_rate*step)
            if explore_p>np.random.rand():
                action = env.action_space.sample()
                
            else: 
                feed = {mainQN.inputs_:state.reshape((1,84,84,4))}
                Qs = sess.run(mainQN.output,feed_dict=feed)
                action = np.argmax(Qs)
            next_state,reward,done,_ = StateProcess(env,action)
            total_reward+=reward
            if done:
                next_state=np.zeros(state.shape)
                t= 1
                print('Episode: {}'.format(ep),
                      'Total reward: {}'.format(total_reward),
                      'Training loss : {:.4f}'.format(loss))
                reward_list.append((ep,total_reward))
                memory.add((state,action,reward,next_state))
                env.reset()
                action =env.action_space.sample()
                state,reward,done,_ =StateProcess(env,action)
            else: 
                memory.add((state,action,reward,next_state))
                state=next_state
                t =0
            batch = memory.sample(batch_size)
            states = np.array([each[0] for each in batch])
            actions = np.array([each[1] for each in batch])
            rewards= np.array([each[2] for each in batch])
            next_states = np.array([each[3] for each in batch])
            
            target_Qs = sess.run(mainQN.output,feed_dict= 
                                 {mainQN.inputs_:next_states})
            episode_ends = (next_states==np.zeros(states[0].shape)).all(axis=1).all(axis=1).all(axis=1)
            target_Qs[episode_ends]=(0,0,0,0,0,0)
            targets = rewards+gamma*np.amax(target_Qs,axis=1)
            loss,_ = sess.run([mainQN.loss,mainQN.opt],
                              feed_dict={mainQN.inputs_:states,
                                         mainQN.targetQs_:targets,
                                         mainQN.actions_:actions})
    saver.save(sess,"checkpoints/Breakout.ckpt")
            
# In[4]

