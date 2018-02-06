## Reinforcement Learning On Mountain Car and an Atari Game

##### MD Muhaimin Rahman
contact: sezan92[at]gmail[dot]com

In this project I have tried Reinforcement Learning on two Open AI games . One is Mountain Car Game and another is AirRaid, an Atari Game.

For the Mountain Car game , I have used Q Learning ALgorithm . The Agent needs to earn -110 reward . We have achieved the reward by running upto 30000 episodes. For the simple Q learning algorithm, I have taken help from [this blog](https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0) . The Video of Full training is [here](https://youtu.be/hClI5zFJlxI) . Please notice that the video is full training , so it is very large . Skip to last 1-2 minutes to watch the final Result . The Code ```MountainCarQLearning.py``` is the Code for Version 1.

#### Update 06 February, 2018
After almost 6 months, I have noticed some bugs in my original code. I have took help from [Max's Blog](http://178.79.149.207/posts/cartpole-qlearning.html) . His implementation of Q learning is better but quite difficult to understand. I have revised the Q learning Class. My implementation is only 34 lines compared to 91 lines of his code! The file ```MountainCarQLearning2.py``` is the Updated and better version.  The file ``` MountainCarQLearning-v2.ipynb``` is the Jupyter notebook of the same code. The file ```QLearn.py``` is the Q learning Implementation Class. The updated version's video is [here](https://youtu.be/0RlK3RkSdnw)

 The  
![Alt text](https://raw.githubusercontent.com/sezan92/ReinforcementOpenAi/master/Final.png) 

For the AirRaid Game, I have used Deep Q Network with Experience replay . For the Architecture I have used the architecture mentioned [here](http://neuro.cs.ut.ee/demystifying-deep-reinforcement-learning/) . 

![Alt text](https://raw.githubusercontent.com/sezan92/ReinforcementOpenAi/master/A.png) 
