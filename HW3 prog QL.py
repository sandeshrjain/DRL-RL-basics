# -*- coding: utf-8 -*-
"""
Created on Mon Feb 28 20:47:18 2022

@author: Sandesh Jain
"""
import gym 
import collections 
import numpy as np
ENV_NAME = "FrozenLake-v0" #you can pick any other grid-world as well,
                           # e.g., 8x8 frozen lake
GAMMA = 0.9 #Discount rate

TEST_EPSDS = 00  #If you want to return test env reults after every update
TOTAL_EPSDS = 500
TMAX = 500 #Max Episode length that is allowed



class Agent_SARSA: 
    def __init__(self, ALPHA, EPSILON): 
        self.env = gym.make(ENV_NAME, is_slippery=False) 
        self.EPSILON = EPSILON
        self.state = self.env.reset() 
        self.values = np.zeros((self.env.nS, self.env.nA))
        self.policy = np.zeros((self.env.nS, 1))
        self.reward = 0
        self.action = self.choose_action(self.state)
        self.ALPHA = ALPHA
        
    def extract_policy(self, value_table, gamma = 1.0): #not used but is kept
        """ 
                Inputs: 
                - value_table: state value function 
            - gamma: discount factor 
             
            Returns: 
                - policy: the optimal policy 
            . 
        """ 
        policy = np.zeros((len(value_table), 1))
        for s in range(self.env.nS):
            a_vals = []
            for a in range(self.env.nA):
                q_sa = 0
                transition = self.env.P[s][a]
                for p, s_, r, t in transition:                       
                    q_sa += p*(r + gamma*value_table[s_])
                a_vals.append(q_sa)  
            max_a = np.argmax(np.asarray(a_vals))
            policy[s] = max_a
                 
        return policy
    
    def sample_env(self): 
        """ 
              Inputs: 
        - self: an agent 
         
              Returns: 
         
        - a tuple: (new_reward, new_state, t) 
              """ 
     
        transition = self.env.P[self.state][self.action]
        transition = transition[0]
        new_reward = transition[2]
        new_state = transition[1]
        t = transition[3]
        self.reward = self.reward + new_reward
        
        return new_reward, new_state, t
        
     
    def choose_action(self, state): 
        """ 
              Inputs: 
        - self: an agent 
        - state: current state 
         
              Returns: 
        - next_a: the next action taken. 
              """ 
              
        p = np.random.random()
        if p< self.EPSILON:
            if np.sum(self.values[state, :]) == 0:
                next_a = np.random.randint(0,self.env.nA)
            else:
                next_a = np.argmax(self.values[state, :])
        else:
            next_a = np.random.randint(0,self.env.nA)
        return next_a
        
     
    def value_update(self, s, a, r, next_s, next_a): 
        """ 
              Inputs: 
        - self: an agent 
        - s: state 
        - a: action 
        - r: reward 
        - next_s: next state 
         
              Returns: 
        - self.values[(s, a)]: the updated value of (s, a). 
              """ 
        self.values[s, a] += self.ALPHA*(r + GAMMA*self.values[next_s, next_a] - 
                                      self.values[s, a])
     
    def play_episode(self, env): 
        """ 
              Inputs: 
        - self: an agent 
        - env: the environment 
         
              Returns: 
        - total_reward: the total reward after playing an   
        episode 
        """ 
        state = env.reset()
        action  = self.choose_action(state)
        reward = 0
    
        for _ in range(TMAX):
            transition = env.P[state][action]
            transition = transition[0]
            new_reward = transition[2]
            new_state = transition[1]
            t = transition[3]
            new_action  = self.choose_action(new_state)
            state = new_state
            action = new_action
            reward += new_reward
            
            if t:
                break
        return reward
            
            
            
    def train(self): 
        """ 
              Inputs: 
        - self: an agent 
        - env: the environment 
         
              Returns: 
        - test_rew_list: the reward list obtained in test env after each update
        - rew_list: List of total rewards after each episode
        - episode_length_list: Length of episodes
        """ 
        test_rew_list = []
        rew_list = []
        episode_length_list = []
        for episodes in range(TOTAL_EPSDS):
            self.state = self.env.reset()
            self.action  = self.choose_action(self.state)
            self.reward = 0
            test_rew_avg = []
            for termi_time in range(TMAX):
                r, s_, t = self.sample_env()
                #r-=0.5
                self.reward+= r
                
                a_ = self.choose_action(s_)
                self.value_update(self.state, self.action, r, s_, a_)
                self.action = a_
                self.state = s_
                if t:
                    break
            episode_length_list.append(termi_time)
            rew_list.append(self.reward/2)    
            for _ in range(TEST_EPSDS):
                test_env = gym.make(ENV_NAME, is_slippery=False) 
                test_rew = self.play_episode(test_env)
                test_rew_avg.append(test_rew)
            test_rew_list.append(np.mean(test_rew_avg))
                    
                    
        #print(self.ALPHA)
        return test_rew_list, rew_list, episode_length_list
               
    
class Agent_QLearning: 
    def __init__(self, alpha, epsilon): 
      self.EPSILON = epsilon

      self.env = gym.make(ENV_NAME, is_slippery=False) 
      self.state = self.env.reset() 
      self.values = np.zeros((self.env.nS, self.env.nA))
      self.policy = np.zeros((self.env.nS, 1))
      self.reward = 0
      self.action = self.choose_action(self.state)
      self.ALPHA = alpha
    def extract_policy(self, value_table, gamma = 1.0): 
        """ 
                Inputs: 
                - value_table: state value function 
            - gamma: discount factor 
             
            Returns: 
                - policy: the optimal policy 
            . 
        """ 
        policy = np.zeros((len(value_table), 1))
        for s in range(self.env.nS):
            a_vals = []
            for a in range(self.env.nA):
                q_sa = 0
                transition = self.env.P[s][a]
                for p, s_, r, t in transition:                       
                    q_sa += p*(r + gamma*value_table[s_])
                a_vals.append(q_sa)  
            max_a = np.argmax(np.asarray(a_vals))
            policy[s] = max_a
                 
        return policy
    
    def sample_env(self): 
        """ 
              Inputs: 
        - self: an agent 
         
              Returns: 
         
        - a tuple: (new_reward, new_state, t) 
              """ 
     
        transition = self.env.P[self.state][self.action]
        transition = transition[0]
        new_reward = transition[2]
        new_state = transition[1]
        t = transition[3]
        #self.reward = self.reward + new_reward
        
        return new_reward, new_state, t
        
     
    def choose_action(self, state): 
        """ 
              Inputs: 
        - self: an agent 
        - state: current state 
         
              Returns: 
        - next_a: the next action taken. 
              """ 
              
        p = np.random.random()
        if p< self.EPSILON:
            if np.sum(self.values[state, :]) == 0:
                next_a = np.random.randint(0,self.env.nA)
            else:
                next_a = np.argmax(self.values[state, :])
        else:
            next_a = np.random.randint(0,self.env.nA)
        return next_a
        
     
    def value_update(self, s, a, r, next_s): 
        """ 
              Inputs: 
        - self: an agent 
        - s: state 
        - a: action 
        - r: reward 
        - next_s: next state 
         
              Returns: 
        - self.values[(s, a)]: the updated value of (s, a). 
              """ 
        self.values[s, a] += self.ALPHA*(r + GAMMA*np.max(self.values[next_s, :]) - 
                                      self.values[s, a])
     
    def play_episode(self, env): 
        """ 
              Inputs: 
        - self: an agent 
        - env: the environment 
         
              Returns: 
        - total_reward: the total reward after playing an   
        episode 
        """ 
        state = env.reset()
        action  = self.choose_action(state)
        reward = 0
    
        for _ in range(TMAX):
            transition = env.P[state][action]
            transition = transition[0]
            new_reward = transition[2]
            new_state = transition[1]
            t = transition[3]
            new_action  = self.choose_action(new_state)
            state = new_state
            action = new_action
            reward += new_reward
            
            if t:
                break
        return reward
            
            
    def train(self): 
        """ 
              Inputs: 
        - self: an agent 
        - env: the environment 
         
              Returns: 
        - test_rew_list: the reward list obtained in test env after each update
        - rew_list: List of total rewards after each episode
        - episode_length_list: Length of episodes
        """ 
        test_rew_list = []
        rew_list = []
        episode_length_list = []
        for episodes in range(TOTAL_EPSDS):
            self.state = self.env.reset()
            self.action  = self.choose_action(self.state)
            self.reward = 0
            test_rew_avg = []
            for termi_time in range(TMAX):
                s_, r, t, _ = self.env.step(self.action)
                self.reward+= r

                a_ = self.choose_action(s_)
                self.value_update(self.state, self.action, r, s_)
                self.action = a_
                self.state = s_
                if t:
                    break
            episode_length_list.append(termi_time)
            rew_list.append(self.reward)    
            for _ in range(TEST_EPSDS):
                test_env = gym.make(ENV_NAME, is_slippery=False) 
                test_rew = self.play_episode(test_env)
                test_rew_avg.append(test_rew)
            test_rew_list.append(np.mean(test_rew_avg))
                    
                    
        #print(self.ALPHA)
        return test_rew_list, rew_list, episode_length_list
    
    
# s_agent = Agent_SARSA()                
# rt_s, r_s = s_agent.train()
import matplotlib.pyplot as plt
#plt.plot(np.arange(len(r_s)), r_s)

# Smoothing function
def smooth(scalars, weight):  # Weight between 0 and 1
    # (From https://stackoverflow.com/questions/42281844/what-is-the-mathematics-behind-the-smoothing-parameter-in-tensorboards-scalar)
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed




# Show the plots for Episode Length for QL v/s SARSA
plt.clf()

s_agent = Agent_SARSA(0.1, 0.9)  
#s_agent.eps = epsilon              
rt_s,_, r_s = s_agent.train()
smooth_rewards_sar = smooth(r_s, 0.9)

q_agent = Agent_QLearning(0.1, 0.9)
rt_q,_, r_q = q_agent.train()
smooth_rewards_q = smooth(r_q, 0.9)

plt.plot(smooth_rewards_sar)

plt.plot(smooth_rewards_q)
plt.title("Q-Learning v/s SARSA episode length")
plt.ylabel('Episode Length')
plt.xlabel('episodes')
plt.legend(['Sarsa', 'Q-Learning'], loc='lower right')
plt.show()



# Show the plots for stepsize for QL v/s SARSA


plt.clf()

for a in [0.01, 0.1, 0.9]:
    alpha= a
    #epsilon = 0.9
    s_agent = Agent_SARSA(alpha, 0.9)  
    #s_agent.eps = epsilon              
    rt_s, r_s,_ = s_agent.train()
    smooth_rewards = smooth(r_s, 0.9)
    plt.plot(smooth_rewards)
    plt.title("SARSA Episodic rewards for different Alphas")
    plt.ylabel('reward')
    plt.xlabel('episodes')
plt.legend(['alpha = 0.01', 'alpha = 0.1', 'alpha = 0.9'], loc='lower right')
