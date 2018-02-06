import numpy as np
class QL:
   def __init__(self,Q,policy,
                legal_actions,
                actions,
                gamma,
                lr):
       self.Q = Q #Q matrix
       self.policy =policy
       self.legal_actions=legal_actions
       self.actions = actions
       self.gamma =gamma
       self.lr =lr
       
   def q_value(self,s,a):
       if (s,a) in self.Q:
           self.Q[(s,a)]
       else:
           self.Q[s,a]=0
       return self.Q[s,a]
   def action(self,s):
       if s in self.policy:
           return self.policy[s]
       else:
           self.policy[s] = np.random.randint(0,self.legal_actions)
       return self.policy[s]
   def learn(self,s,a,s1,r,done):
       if done== False:
           self.Q[(s,a)] =self.q_value(s,a)+ self.lr*(r+self.gamma*max([self.q_value(s1,a1) for a1 in self.actions]) - self.q_value(s,a))
       else:
           self.Q[(s,a)] =self.q_value(s,a)+ self.lr*(r - self.q_value(s,a))
       self.q_values = [self.q_value(s,a1) for a1 in self.actions]
       self.policy[s] = self.actions[self.q_values.index(max(self.q_values))]
   