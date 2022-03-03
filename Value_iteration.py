import gym 
import numpy as np 
 
# make the frozen lake environment using OpenAI’s Gym 
env = gym.make("FrozenLake-v0", is_slippery=False) # or the latest version 
 
# explore the environment 
print(env.observation_space.n) 

print(env.action_space.n) 
"""
    - 0: LEFT
    - 1: DOWN
    - 2: RIGHT
    - 3: UP
"""
def pretty_print(pi):
    print("vals")
    for i in range(0, len(pi),4):
        print(pi[i:i+4].T)
max_val = 7        
def value_iteration(env, gamma = 1.0): 
    """     
    Inputs: 
- env: the frozen lake environment. 
- gamma: discount factor 
 
    Returns: 
- value_table: state value function 
- Q_value: state-action value function (Q function) 
- policy: Extracted policy
- value_history: Extracted Value history over iterations
    """ 
    value_history = np.zeros((max_val, env.observation_space.n))
    value_table_old = np.zeros((env.observation_space.n,1))
    value_table = np.zeros((env.observation_space.n,1))
    Q_value = np.zeros((env.observation_space.n,env.action_space.n))
    policy = np.zeros((len(value_table), 1))
    j = -1
    while True:
        j+=1
        for s in range(env.nS):
            a_vals = []
            for a in range(env.nA):
                q_sa = 0
                transition = env.P[s][a]
                for p, s_, r, t in transition:                       
                    q_sa += p*(r + gamma*value_table_old[s_])
                a_vals.append(q_sa)  
                Q_value[s][a] = q_sa
            max_a = np.argmax(np.asarray(a_vals))
            value_table[s] = a_vals[max_a]
            value_history[j, s] = a_vals[max_a]
            policy[s] = max_a
        pretty_print(value_table)
        if j>max_val-2:
            break
        else:
            value_table_old  = np.copy(value_table)
    return value_table, Q_value, policy, value_history
                    
                
gamma = 0.9
v, q, pi_star, vh = value_iteration(env, gamma)                
            
            
def extract_policy(value_table, gamma = 1.0): 
    """ 
    Inputs: 
    - value_table: state value function 
- gamma: discount factor 
 
Returns: 
    - policy: the optimal policy 
. 
    """ 
    policy = np.zeros((len(value_table), 1))
    for s in range(env.nS):
        a_vals = []
        for a in range(env.nA):
            q_sa = 0
            transition = env.P[s][a]
            for p, s_, r, t in transition:                       
                q_sa += p*(r + gamma*value_table[s_])
            a_vals.append(q_sa)  
        max_a = np.argmax(np.asarray(a_vals))
        policy[s] = max_a
        
    return policy
    
pi = extract_policy(v, gamma) 
    
    

        
import matplotlib.pyplot as plt

plt.plot(vh, label=['v0', 'v1', 'v2'])
plt.xlabel("iterations")
plt.ylabel("Value")
plt.title("value v/s iterations")    
