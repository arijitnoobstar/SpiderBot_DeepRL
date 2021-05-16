#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# SpiderBot_Replay_Buffer.py                                     #
# Author(s): Chong Yu Quan, Arijit Dasgupta                   #
# Email(s): chong.yuquan@u.nus.edu, arijit.dasgupta@u.nus.edu #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

""" 
This source code file contains the code for the replay_buffer class
Purpose 1: store memory of state, action, state_prime, reward, terminal flag 
Purpose 2: function to randomly sample a batch of memory
"""

# Standard Imports
import numpy as np

class replay_buffer:
    
    def __init__(self, max_mem_size, state_input_shape, action_space):
        
        """ class constructor that initialises memory states attributes """
        
        # bound for memory log
        self.mem_size = max_mem_size
        
        # counter for memory logged
        self.mem_counter = 0 
        
        # logs for state, action, state_prime, reward, terminal flag
        self.state_log = np.zeros((self.mem_size, *state_input_shape))
        self.state_prime_log = np.zeros((self.mem_size, *state_input_shape))
        
        if action_space != 1:
            
            self.action_log = np.zeros((self.mem_size, action_space))
        
        else:
            
            self.action_log = np.zeros(self.mem_size)
            
        self.reward_log = np.zeros(self.mem_size)
        self.terminal_log = np.zeros(self.mem_size)
        
    def log(self, state, action, reward, state_prime, is_done):
        
        """ log memory """
        
        # index for logging. based on first in first out
        index = self.mem_counter % self.mem_size
        
        # log memory for state, action, state_prime, reward, terminal flag
        self.state_log[index] = state
        self.state_prime_log[index] = state_prime
        self.action_log[index] = action
        self.reward_log[index] = reward
        self.terminal_log[index] = is_done

        # increment counter
        self.mem_counter += 1
    
    def sample_log(self, batch_size):
        
        """ function to randomly sample a batch of memory """
        
        # select amongst memory logs that is filled
        max_mem = min(self.mem_counter, self.mem_size)
        
        # randomly select memory from logs
        batch = np.random.choice(max_mem, batch_size, replace = False)
        
        # obtain corresponding state, action, state_prime, reward, terminal flag
        states = self.state_log[batch]
        states_prime = self.state_prime_log[batch]
        actions = self.action_log[batch]
        rewards = self.reward_log[batch]
        is_dones = self.terminal_log[batch]
        
        return states, actions, rewards, states_prime, is_dones