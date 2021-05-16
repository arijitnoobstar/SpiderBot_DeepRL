#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# SpiderBot_Agent.py                                             #
# Author(s): Chong Yu Quan, Arijit Dasgupta                   #
# Email(s): chong.yuquan@u.nus.edu, arijit.dasgupta@u.nus.edu #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# Standard Imports
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import os
# SpiderBot Imports
from SpiderBot_Neural_Network import fc_model
from SpiderBot_Replay_Buffer import replay_buffer 

""" 
Agent class 
Purpose 1 : creates and updates neural network 
Purpose 2 : processes output from neural network to decide action for p_gym
5 Algorithms available: MAD3QN, MAA2C, A2CMA, A2CSA, DDPG
"""

class Agent:
    
    def __init__(self, model, num_of_legs, num_of_joints, discount_rate, lr_actor, lr_critic, action_space, 
                 tau, max_mem_size, batch_size, noise, max_action, min_action, epsilon, epsilon_decay, epsilon_min,
                 update_target, training_name):
        
        """ class constructor that initialises discount rate for critic loss, learning rate for actor and critic """
        """ as well as the neural network models for actor and critic """
        
        self.model = model
        
        # number of legs
        self.num_of_legs = num_of_legs
        
        # number of joints
        self.num_of_joints = num_of_joints
        
        # discount rate for critic loss (TD error)
        self.discount_rate = discount_rate
        
        # learning rate for actor model
        self.lr_actor = lr_actor
        
        # learning rate for critic model
        self.lr_critic = lr_critic
        
        # stores action space
        self.action_space = action_space
        
        # for two seperate actor and critic model
        if self.model == "MAA2C": 
            
            # stores selected actions (tensor) from actors
            self.actions = [0 for x in range(self.num_of_legs)]
            
            # stores all actors
            self.MAA2C_actors_list = [0 for x in range(self.num_of_legs)]
            
            # iterate over each leg in spiderbot to generate one actor model
            # each actor takes in localised state observation, i.e (60,), for each leg and outputs softmax of _ "actions"
            # each "action" is a combination of _ joint positions in degrees for 4 joints
            for x in range(1, self.num_of_legs + 1, 1):

                # creates actor models and append to actor_list 
                self.MAA2C_actors_list[x-1] = fc_model(model = "MAA2C_Actor", num_of_legs = self.num_of_legs, 
                                                 num_of_joints = self.num_of_joints,
                                                 h_units = [2048, 1024, 512], weight_decay = [0, 0, 0], 
                                                 dropout_rate = [0, 0, 0], num_of_outputs = self.action_space, 
                                                 training_name = training_name)

                # update actor model_names attributes for checkpoints
                self.MAA2C_actors_list[x-1].model_name = "MAA2C_Actor_" + str(x)

                # update actor checkpoints_path attributes
                self.MAA2C_actors_list[x-1].checkpoint_path = os.path.join(self.MAA2C_actors_list[x-1].checkpoint_dir, 
                                                                    self.MAA2C_actors_list[x-1].model_name)

                # compile actor models using Adam optimiser with respective learning rate
                self.MAA2C_actors_list[x-1].compile(optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr_actor))

            # creates critic model. critic has access to whole state observation, i.e. (493,)
            # outputs state value, V, for a given state
            self.MAA2C_Critic = fc_model(model = "MAA2C_Critic", num_of_legs = self.num_of_legs, num_of_joints = self.num_of_joints,
                                   h_units = [512, 256, 128], weight_decay = [0, 0, 0], dropout_rate = [0, 0, 0], 
                                   num_of_outputs = 1, training_name = training_name)

            # update critic model_names attributes for checkpoints
            self.MAA2C_Critic.model_name = "MAA2C_Critic"

            # update critic checkpoints_path attributes
            self.MAA2C_Critic.checkpoint_path = os.path.join(self.MAA2C_Critic.checkpoint_dir, self.MAA2C_Critic.model_name)

            # compile critic model using Adam optimiser with respective learning rate
            self.MAA2C_Critic.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr_critic))
        
        # for one combined hybrid actor critic model
        elif self.model == "A2C_MultiAction": 
            
            # stores selected actions (tensor) from actors
            self.actions = [0 for x in range(self.num_of_legs)]
            
            # creates actor_critic model
            self.A2C_MultiAction = fc_model(model = "A2C_MultiAction", num_of_legs = self.num_of_legs, 
                                            num_of_joints = self.num_of_joints,
                                            h_units = [2048, 1024, 512], weight_decay = [0, 0, 0], 
                                            dropout_rate = [0, 0, 0], num_of_outputs = self.action_space, 
                                            training_name = training_name)
            
            # update actor_critic model_names attributes for checkpoints
            self.A2C_MultiAction.model_name = "A2C_MultiAction"

            # update actor_critic checkpoints_path attributes
            self.A2C_MultiAction.checkpoint_path = os.path.join(self.A2C_MultiAction.checkpoint_dir, 
                                                                self.A2C_MultiAction.model_name)

            # compile actor_critic model using Adam optimiser with learning rate of actor
            self.A2C_MultiAction.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr_actor))
        
        # for one combined hybrid actor critic model
        elif self.model == "A2C_SingleAction": 
            
            # stores selected actions (tensor) from actors
            self.actions = [0 for x in range(2)]
            
            # creates actor_critic model
            self.A2C_SingleAction = fc_model(model = "A2C_SingleAction", num_of_legs = self.num_of_legs, 
                                            num_of_joints = self.num_of_joints, h_units = [512, 256, 128], 
                                            weight_decay = [0, 0, 0], dropout_rate = [0, 0, 0], 
                                            num_of_outputs = self.action_space, training_name = training_name)
            
            # update actor_critic model_names attributes for checkpoints
            self.A2C_SingleAction.model_name = "A2C_SingleAction"

            # update actor_critic checkpoints_path attributes
            self.A2C_SingleAction.checkpoint_path = os.path.join(self.A2C_SingleAction.checkpoint_dir, 
                                                                self.A2C_SingleAction.model_name)

            # compile actor_critic model using Adam optimiser with learning rate of actor
            self.A2C_SingleAction.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr_actor))
            
        # deep deterministic policy gradient
        elif self.model == "DDPG":
            
            # softcopy parameter for target network 
            self.tau = tau

            # counter for apply gradients
            self.apply_grad_counter = 0 
            # step for apply_grad_counter to hardcopy weights of original to target
            self.update_target = update_target
            
            # memory for replay
            self.memory = replay_buffer(max_mem_size, [13 + self.num_of_joints*15], self.num_of_joints)
            
            # batch of memory to sample
            self.batch_size = batch_size
            
            # noise for action
            self.noise = noise
            
            # upper and lower bounds for actions
            self.max_action = max_action
            self.min_action = min_action
            
            # intialise actor model
            self.DDPG_Actor = fc_model(model = "DDPG_Actor", num_of_legs = self.num_of_legs, num_of_joints = self.num_of_joints,
                                  h_units = [512, 256, 128], 
                                  weight_decay = [0, 0, 0], dropout_rate = [0, 0, 0], 
                                  num_of_outputs = self.num_of_joints, training_name = training_name)
            
            # update actor model_names attributes for checkpoints
            self.DDPG_Actor.model_name = "DDPG_Actor"

            # update actor checkpoints_path attributes
            self.DDPG_Actor.checkpoint_path = os.path.join(self.DDPG_Actor.checkpoint_dir, self.DDPG_Actor.model_name)
            
            # intialise target actor model
            self.DDPG_Target_Actor = fc_model(model = "DDPG_Actor", num_of_legs = self.num_of_legs, num_of_joints = self.num_of_joints,
                                         h_units = [512, 256, 128], 
                                         weight_decay = [0, 0, 0], dropout_rate = [0, 0, 0], 
                                         num_of_outputs = self.num_of_joints, training_name = training_name)
            
            # update target actor model_names attributes for checkpoints
            self.DDPG_Target_Actor.model_name = "DDPG_Target_Actor"

            # update target actor checkpoints_path attributes
            self.DDPG_Target_Actor.checkpoint_path = os.path.join(self.DDPG_Target_Actor.checkpoint_dir, self.DDPG_Target_Actor.model_name)
            
            # intialise critic model
            self.DDPG_Critic = fc_model(model = "DDPG_Critic", num_of_legs = self.num_of_legs, num_of_joints = self.num_of_joints, 
                                   h_units = [512, 256, 128], 
                                   weight_decay = [0, 0, 0], dropout_rate = [0, 0, 0], num_of_outputs = 1, training_name = training_name)

            # update critic model_names attributes for checkpoints
            self.DDPG_Critic.model_name = "DDPG_Critic"

            # update critic checkpoints_path attributes
            self.DDPG_Critic.checkpoint_path = os.path.join(self.DDPG_Critic.checkpoint_dir, self.DDPG_Critic.model_name)
            
            # intialise target critic model
            self.DDPG_Target_Critic = fc_model(model = "DDPG_Critic", num_of_legs = self.num_of_legs, num_of_joints = self.num_of_joints, 
                                          h_units = [512, 256, 128], 
                                          weight_decay = [0, 0, 0], dropout_rate = [0, 0, 0], num_of_outputs = 1, training_name = training_name)

            # update target critic model_names attributes for checkpoints
            self.DDPG_Target_Critic.model_name = "DDPG_Target_Critic"

            # update target critic checkpoints_path attributes
            self.DDPG_Target_Critic.checkpoint_path = os.path.join(self.DDPG_Target_Critic.checkpoint_dir, 
                                                              self.DDPG_Target_Critic.model_name)
            
            # compile actor, target_actor, critic, target_critic
            self.DDPG_Actor.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr_actor))
            self.DDPG_Target_Actor.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr_actor))
            self.DDPG_Critic.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr_critic))
            self.DDPG_Target_Critic.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr_critic))
            
            # hard update target models' weights to online network to match initialised weights
            self.update_ddpg_target_models(tau = 1)
        
        # multi-agent dueling double dqn (mad3qn)
        elif self.model == "MAD3QN":

            # softcopy parameter for target network 
            self.tau = tau
            
            # list of possible actions
            self.actions_list = [x for x in range(self.action_space)]
            
            # exploration constant
            self.epsilon = epsilon
            
            # decay for exploration constant 
            self.epsilon_decay = epsilon_decay
            
            # minimum exploration constant
            self.epsilon_min = epsilon_min
            
            # batch of memory to sample
            self.batch_size = batch_size
            
            # counter for apply gradients
            self.apply_grad_counter = 0 
            
            # step for apply_grad_counter to hardcopy weights of original to target
            self.update_target = update_target
            
            # memory for replay
            self.memory = replay_buffer(max_mem_size, [13 + self.num_of_joints*15], self.num_of_legs)
            
            # stores all list of q_eval and q_target models 
            self.q_eval_list = [0 for x in range(self.num_of_legs)]
            self.q_target_list = [0 for x in range(self.num_of_legs)]
            
            # iterate over each leg in spiderbot to generate one q_eval and q_target model each
            # each q_eval takes in global state observation, i.e (493,), for each leg and outputs Q value of _ "actions"
            # each "action" is a combination of _ joint positions in degrees for 4 joints
            for x in range(1, self.num_of_legs + 1, 1):

                # creates q_eval models and append to q_eval_list 
                self.q_eval_list[x-1] = fc_model(model = "MAD3QN", num_of_legs = self.num_of_legs, 
                                                 num_of_joints = self.num_of_joints,
                                                 h_units = [1024, 512, 512], weight_decay = [0, 0, 0], 
                                                 dropout_rate = [0, 0, 0], num_of_outputs = self.action_space, training_name = training_name)

                # update q_eval model_names attributes for checkpoints
                self.q_eval_list[x-1].model_name = "q_eval_" + str(x)

                # update q_eval checkpoints_path attributes
                self.q_eval_list[x-1].checkpoint_path = os.path.join(self.q_eval_list[x-1].checkpoint_dir, 
                                                                     self.q_eval_list[x-1].model_name)

                # compile q_eval models using Adam optimiser with respective learning rate and loss as MSE
                self.q_eval_list[x-1].compile(optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr_actor), 
                                             loss = "mean_squared_error")
                
                # creates q_target models and append to q_target_list 
                self.q_target_list[x-1] = fc_model(model = "MAD3QN", num_of_legs = self.num_of_legs, 
                                                   num_of_joints = self.num_of_joints,
                                                   h_units = [1024, 512, 512], weight_decay = [0, 0, 0], 
                                                   dropout_rate = [0, 0, 0], num_of_outputs = self.action_space, training_name = training_name)

                # update q_target model_names attributes for checkpoints
                self.q_target_list[x-1].model_name = "q_target_" + str(x)

                # update q_target checkpoints_path attributes
                self.q_target_list[x-1].checkpoint_path = os.path.join(self.q_target_list[x-1].checkpoint_dir, 
                                                                       self.q_target_list[x-1].model_name)

                # compile q_target models using Adam optimiser with respective learning rate and loss as MSE
                self.q_target_list[x-1].compile(optimizer = tf.keras.optimizers.Adam(learning_rate = self.lr_actor), 
                                                loss = "mean_squared_error")
            # hard update target network weights to online network to match initialised weights
            self.update_mad3qn_target_model(tau = 1)

                
    def update_ddpg_target_models(self, tau = None): 
        
        """ function to soft update target model weights for DDPG. Hard update is possible if tau = 1 """
        
        # use tau attribute if tau not specified 
        if tau is None:
            
            tau = self.tau
        
        # weight list to store processed target actor weights
        weights = []
        
        # target actor weights
        targets = self.DDPG_Target_Actor.weights
        
        # enumerate over current actors weights
        for i, weight in enumerate(self.DDPG_Actor.weights):
            
            # softcopy of actors weight
            weights.append(weight * tau + targets[i] * (1 - tau))
        
        # append processed weights to target actor 
        self.DDPG_Target_Actor.set_weights(weights)
        
        # weight list to store processed target critic weights
        weights = []
        
        # target critic weights
        targets = self.DDPG_Target_Critic.weights
        
        # enumerate over current critic weights
        for i, weight in enumerate(self.DDPG_Critic.weights):
            
             # softcopy of critic weight
            weights.append(weight * tau + targets[i] * (1 - tau))
        
        # replace processed weights for target critic
        self.DDPG_Target_Critic.set_weights(weights)

    def update_mad3qn_target_model(self, tau = None): 
        
        """ function to soft update target model weights for mad3qn. Hard update is possible if tau = 1 """
        
        # use tau attribute if tau not specified 
        if tau is None:
            
            tau = self.tau

        # Loop 
        for x in range(1, self.num_of_legs + 1, 1):

            weights = []
            
            # target network weights
            targets = self.q_target_list[x-1].weights
            
            # enumerate over online network weights
            for i, weight in enumerate(self.q_eval_list[x-1].weights):
                
                # softcopy of online network weight
                weights.append(weight * tau + targets[i] * (1 - tau))
            
            # replace processed weights for target network
            self.q_target_list[x-1].set_weights(weights)
        
    
    def store_memory(self, state, action, reward, state_prime, is_done):
    
        """ function to log state, action, state_prime, reward, terminal flag """

            
        self.memory.log(state, action, reward, state_prime, is_done)
        
        
    def select_actions(self, observations, mode):
        
        """ function to select actions for each leg from observations from return_reward_obv_leg """
        """ observations should be a (60, 8) numpy array """
        
        # list to return actions in numpy 
        actions_list = []
        
        # for two seperate actor and critic model
        if self.model == "MAA2C":
            
            # iterate over observations from each leg
            for x in range(self.num_of_legs):

                # convert observations into tensor
                state = tf.convert_to_tensor([observations[x]], dtype = tf.float32)

                # feed observation tensor to corresponding actor model to obtain softmax probabilities
                probs = self.MAA2C_actors_list[x](state)
                
                # convert tensor to numpy array
                probs = probs.numpy()[0]
                
                # replace any NaN values to 0 if any
                probs = np.nan_to_num(probs)
                
                # action is the index that has the largest probability
                action = np.argmax(probs)
                
                # append action to action_list (numpy)
                actions_list.append(action)
                
                # store action (tensor) in self.actions
                self.actions[x] = tf.convert_to_tensor([action], dtype = tf.float32)
                
        # for combined hybrid actor critic model (v1)
        elif self.model == 'A2C_MultiAction':
            
            # convert observations into tensor
            state = tf.convert_to_tensor([observations], dtype = tf.float32)

            # feed observation tensor to corresponding actor model to obtain list of softmax probabilities
            _, probs = self.A2C_MultiAction(state)
            
            # iterate over pdf from each leg
            for x in range(self.num_of_legs):
                
                # convert tensor corresponding to specific leg to numpy array
                prob = probs[x].numpy()[0]
                
                # replace any NaN values to 0 if any
                prob = np.nan_to_num(prob)
                
                # action is the index that has the largest probability
                action = np.argmax(prob)
                
                # append action to action_list (numpy)
                actions_list.append(action)
                
                # store action (tensor) in self.actions
                self.actions[x] = tf.convert_to_tensor([action], dtype = tf.float32)
        
        # for combined hybrid actor critic model (v2)
        elif self.model == 'A2C_SingleAction':
            
            # convert observations into tensor
            state = tf.convert_to_tensor([observations], dtype = tf.float32)

            # feed observation tensor to corresponding actor model to obtain softmax probabilities
            # pdf for joint and pdf for action for joint 
            _, probs = self.A2C_SingleAction(state)
                
            # iterate over pdf of selected joint and selected action
            for x in range(2):
                
                # convert tensor to numpy array
                prob = probs[x].numpy()[0]
                
                # replace any NaN values to 0 if any
                prob = np.nan_to_num(prob)
                
                # action is the index that has the largest probability
                action = np.argmax(prob)
                
                # append action to action_list (numpy)
                actions_list.append(action)
                
                # store action (tensor) in self.actions
                self.actions[x] = tf.convert_to_tensor([action], dtype = tf.float32)

        # for ddpg
        elif self.model == "DDPG":
        
            # convert observations into tensor
            state = tf.convert_to_tensor([observations], dtype = tf.float32)
            
            # feed observation tensor to actor model to obtain list of bounded actions (tanh --> +-1)
            actions = self.DDPG_Actor(state)
            
            # increase bound to range of max_action (e.g. +- 10)
            actions = actions * self.max_action
            
            # add gaussian noise if not test
            if mode != "test":
                
                actions += tf.random.normal(shape = [self.num_of_joints], mean = 0.0, stddev = self.noise)
            
            # ensure actions are within range 
            actions = tf.clip_by_value(actions, self.min_action, self.max_action)
            
            return actions[0] 
        
        # for mad3qn
        elif self.model == "MAD3QN":
            
            # iterate over each leg
            for x in range(self.num_of_legs): 
                
                # select action randomly for exploration
                if np.random.random() < self.epsilon and mode != "test":

                    action = np.random.choice(self.actions_list)
                
                # select action greedily for exploitation
                else:

                    # convert observations into tensor
                    state = tf.convert_to_tensor([observations], dtype = tf.float32)

                    # feed observation tensor to actor model to obtain actions
                    actions = self.q_eval_list[x](state)
                    
                    # obtain action with largest Q
                    action = tf.math.argmax(actions, axis = 1).numpy()[0]
            
                # append action to action_list (numpy)
                actions_list.append(action)
                    
        return actions_list
    
    def apply_gradients_MAD3QN(self):
        
        """ function to apply gradients for mad3qn """
        """ learns from replay buffer """

        # doesnt not apply gradients if memory does not have at least batch_size number of logs
        if self.memory.mem_counter < self.batch_size:    
            return
    
        # sample batch of memory of state, action, state_prime, reward, terminal flag from memory log
        states, actions, rewards, states_prime, is_done = self.memory.sample_log(self.batch_size)
            
        # losses list
        losses = []
        
        # iterate over each leg
        for x in range(self.num_of_legs): 
            
            # compute q values of current state using eval model 
            q = self.q_eval_list[x](states)
            
            # compute q values of next state using target model
            q_prime = self.q_target_list[x](states_prime)
            
            # obtain numpy copy of q values of next state using eval model
            q_target = q.numpy()
            
            # obtain maximal actions from current state using eval model 
            max_actions = tf.math.argmax(self.q_eval_list[x](states), axis = 1)
            
            # enumerate over is_done array
            for index, terminate in enumerate(is_done):
                
                # for each q value for a given current state and action sampled based on q_eval (online), 
                # q_target = reward + discount_rate * (q values of (state, maximal actions) from ***Q_TARGET***)
                q_target[index, int(actions[index][x])] = rewards[index] + self.discount_rate * q_prime[index, max_actions[index]] * (1 - is_done[index])
            
            # train on batch size of memory 
            losses.append(self.q_eval_list[x].train_on_batch(states, q_target))
            
        # if exploration constant greater than minimum
        if self.epsilon > self.epsilon_min:
            
            # decay
            self.epsilon = self.epsilon - self.epsilon_decay
        
        # else remain as epsilon_min
        else:
            
            self.epsilon = self.epsilon_min
        
        # increment of apply_grad_counter
        self.apply_grad_counter += 1 

        # SOFT COPY OPTION: update target models based on user specified tau
        if self.update_target == None:

             self.update_mad3qn_target_model()    

        # HARD COPY OPTION EVERY update_target steps
        else:
            if self.apply_grad_counter % self.update_target == 0: 
            
                self.update_mad3qn_target_model(tau = 1)

        # return total losses
        return sum(losses)

    def apply_gradients_DDPG(self):
        
        """ function to apply gradients for ddpg """
        """ learns from replay buffer """
        # doesnt not apply gradients if memory does not have at least batch_size number of logs
        if self.memory.mem_counter < self.batch_size:
            return
        
        # randomly sample batch of memory of state, action, state_prime, reward, terminal flag from memory log
        state, action, reward, state_prime, is_done = self.memory.sample_log(self.batch_size)
        
        # convert state, action, state_prime, reward to tensors
        states = tf.convert_to_tensor(state, dtype = tf.float32)
        states_prime = tf.convert_to_tensor(state_prime, dtype = tf.float32)
        actions = tf.convert_to_tensor(action, dtype = tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype = tf.float32)
        
        # record operations for automatic differentiation for critic 
        with tf.GradientTape(persistent = True) as tape: 
            
            # obtain actions from target actor for states_prime
            target_actions = self.DDPG_Target_Actor(states_prime)
            
            # obtain critic q value by feeding critic with states_prime and target_actions 
            target_critic_value = tf.squeeze(self.DDPG_Target_Critic(tf.concat([states_prime, target_actions], axis = 1)), 
                                             axis = 1)
            
            # obtain critic q value by feeding critic with states and actions 
            critic_value = tf.squeeze(self.DDPG_Critic(tf.concat([states, actions], axis = 1)), axis = 1)
            
            # obtain td target
            td_target = rewards + self.discount_rate * target_critic_value * (1 - is_done)
            
            # critic loss is mean squared error between td_target and critic value 
            critic_loss = tf.keras.losses.MSE(td_target, critic_value)
        
        # computes critic gradient for all trainable variables using operations recorded in context of this tape
        critic_gradient = tape.gradient(critic_loss, self.DDPG_Critic.trainable_variables)
        
        # apply critic gradients to all trainable variables in critic model
        self.DDPG_Critic.optimizer.apply_gradients(zip(critic_gradient, self.DDPG_Critic.trainable_variables))
        
        # record operations for automatic differentiation for actor
        with tf.GradientTape(persistent = True) as tape: 
            
            # obtain actions from state following different policy 
            new_pol_actions = self.DDPG_Actor(states)
            
            # gradient ascent using critic value ouput as actor loss
            # loss is coupled with actor model from new_pol_actions 
            actor_loss = -self.DDPG_Critic(tf.concat([states, new_pol_actions], axis = 1))
            
            # reduce mean across batch_size
            actor_loss = tf.math.reduce_mean(actor_loss)
        
        # computes actor gradient for all trainable variables using operations recorded in context of this tape
        actor_gradient = tape.gradient(actor_loss, self.DDPG_Actor.trainable_variables)
        
        # apply actor gradients to all trainable variables in actor model
        self.DDPG_Actor.optimizer.apply_gradients(zip(actor_gradient, self.DDPG_Actor.trainable_variables))        

        # increment of apply_grad_counter
        self.apply_grad_counter += 1 

        # SOFT COPY OPTION: update target models based on user specified tau
        if self.update_target == None:

             self.update_ddpg_target_models()    

        # HARD COPY OPTION EVERY update_target steps
        else:
            if self.apply_grad_counter % self.update_target == 0: 
            
                self.update_ddpg_target_models(tau = 1)

        # gather total loss for logging
        total_loss = critic_loss + actor_loss

        # return the total loss for logging
        return total_loss.numpy()


    def apply_gradients_MAA2C(self, critic_observations, reward, critic_observations_prime, is_done, actor_observations):
        
        """ function to apply gradients for learning to actors and critic seperately for MAA2C"""
        """ actor's observations should be a (60, 8) numpy array """
        """ crtic's observations should be a (493, ) numpy array if it is 8 legged """
        # return 0
        # list to store transition values in calculating gradients for each actor model
        actor_state_list = [0 for x in range(self.num_of_legs)]
        probs_list = [0 for x in range(self.num_of_legs)]
        action_probs_list = [0 for x in range(self.num_of_legs)]
        log_prob_list = [0 for x in range(self.num_of_legs)]
        actor_loss_list = [0 for x in range(self.num_of_legs)]
        actor_gradients_list = [0 for x in range(self.num_of_legs)]
        
        # convert critic observations into tensor
        critic_state = tf.convert_to_tensor([critic_observations], dtype = tf.float32)
        critic_state_prime = tf.convert_to_tensor([critic_observations_prime], dtype = tf.float32)
        
        # iterate over each actors observations and convert them into tensors
        for x in range(self.num_of_legs):
            
            actor_state_list[x] = tf.convert_to_tensor([actor_observations[x]], dtype = tf.float32)
        
        # convert rewards into tensor 
        reward = tf.convert_to_tensor(reward, dtype = tf.float32)
        
        # record operations for automatic differentiation.
        with tf.GradientTape(persistent = True) as tape:

            # obtain critic state value, V(s), from current observation
            critic_state_value = self.MAA2C_Critic(critic_state)
            
            # obtain critic state value prime, V(s'), from observation after a specified time step 
            critic_state_value_prime = self.MAA2C_Critic(critic_state_prime)
            
            # ensure that state values obtained are scalar
            critic_state_value = tf.squeeze(critic_state_value)
            critic_state_value_prime = tf.squeeze(critic_state_value_prime)
            
            # obtain td error = reward(r) + discount_rate(gamma) * V(s') + V(s)
            td_error = reward + self.discount_rate * critic_state_value_prime * (1 - is_done) - critic_state_value
            
            # calculate critic loss = (td_error)^2
            critic_loss =  td_error**2
            
            # iterate over each actor for each spiderbot leg
            for x in range(self.num_of_legs):
                
                # obtain obtain softmax probabilities from current observation
                probs_list[x] = self.MAA2C_actors_list[x](actor_state_list[x])
                
                # create catergorical distribution 
                action_probs_list[x] = tfp.distributions.Categorical(probs = probs_list[x])
                
                # calculate log probablity of selected actions for a given state 
                log_prob_list[x] = action_probs_list[x].log_prob(self.actions[x])
            
                # calculate actor loss
                actor_loss_list[x] = -log_prob_list[x] * td_error
        
        # computes critic gradient for all trainable variables using operations recorded in context of this tape
        critic_gradients = tape.gradient(critic_loss, self.MAA2C_Critic.trainable_variables)
        
        # apply critic gradients to all trainable variables in critic model
        self.MAA2C_Critic.optimizer.apply_gradients(zip(critic_gradients, self.MAA2C_Critic.trainable_variables))
        
        # iterate over each leg 
        for x in range(self.num_of_legs):
            
            # computes actor gradient for all trainable variables using operations recorded in context of this tape
            actor_gradients_list[x] = tape.gradient(actor_loss_list[x], self.MAA2C_actors_list[x].trainable_variables)
            
            # apply actor gradients to all trainable variables in actor model
            self.MAA2C_actors_list[x].optimizer.apply_gradients(zip(actor_gradients_list[x], 
                                                              self.MAA2C_actors_list[x].trainable_variables))
        
        # delete reference to tape
        del tape

        # gather total loss for logging
        total_loss = critic_loss + sum(actor_loss_list)

        # return the total loss for logging
        return total_loss.numpy()[0]

    def apply_gradients_A2C_MultiAction(self, observations, reward, observations_prime, is_done):
        
        """ function to apply gradients for learning to A2C_MultiAction """
        """ observations should be a (493, ) numpy array if it is 8 legged """
        
        # list to store transition values in calculating gradients for each actor model
        action_probs_list = [0 for x in range(self.num_of_legs)]
        log_prob_list = [0 for x in range(self.num_of_legs)]
        actor_loss_list = [0 for x in range(self.num_of_legs)]
        
        # convert observations into tensor
        state = tf.convert_to_tensor([observations], dtype = tf.float32)
        state_prime = tf.convert_to_tensor([observations_prime], dtype = tf.float32)
        
        # convert rewards into tensor 
        reward = tf.convert_to_tensor(reward, dtype = tf.float32)
        
        # record operations for automatic differentiation.
        with tf.GradientTape() as tape:

            # obtain state value, V(s), and list of softmax probabilities for each leg from current observation
            state_value, probs = self.A2C_MultiAction(state)
            
            # obtain state value prime, V(s'), from observation after a specified time step 
            state_value_prime, _ = self.A2C_MultiAction(state_prime)
            
            # ensure that state values obtained are scalar
            state_value = tf.squeeze(state_value)
            state_value_prime = tf.squeeze(state_value_prime)
            
            # obtain td error = reward(r) + discount_rate(gamma) * V(s') + V(s)
            td_error = reward + self.discount_rate * state_value_prime * (1 - is_done) - state_value
            
            # iterate over each pdf in probabilties list for each spiderbot leg
            for x in range(self.num_of_legs):
                
                # create catergorical distribution from each pdf in probabilties list
                action_probs_list[x] = tfp.distributions.Categorical(probs = probs[x])
                
                # calculate log probablity of selected actions for a given state for each spiderbot leg
                log_prob_list[x] = action_probs_list[x].log_prob(self.actions[x])
            
                # calculate actor loss for each spiderbot leg 
                actor_loss_list[x] = -log_prob_list[x] * td_error
            
            # calculate critic loss = (td_error)^2
            critic_loss =  td_error**2
            
            # sum together all losses of actor critic combined hybrid model
            total_loss = critic_loss + sum(actor_loss_list)
        
        # computes actor critic gradient for all trainable variables using operations recorded in context of this tape
        gradients = tape.gradient(total_loss, self.A2C_MultiAction.trainable_variables)
        
        # apply actor critic gradients to all trainable variables in actor critic model
        self.A2C_MultiAction.optimizer.apply_gradients(zip(gradients, self.A2C_MultiAction.trainable_variables))
        
        # delete reference to tape
        del tape

        # return the total loss for logging
        return total_loss.numpy()[0]
    
    def apply_gradients_A2C_SingleAction(self, observations, reward, observations_prime, is_done):
        
        """ function to apply gradients for learning to A2C_SingleAction """
        """ observations should be a (493, ) numpy array if it is 8 legged """
        
         # list to store transition values in calculating gradients for each actor model
        action_probs_list = [0 for x in range(2)]
        log_prob_list = [0 for x in range(2)]
        actor_loss_list = [0 for x in range(2)]
        
        # convert observations into tensor
        state = tf.convert_to_tensor([observations], dtype = tf.float32)
        state_prime = tf.convert_to_tensor([observations_prime], dtype = tf.float32)
        
        # convert rewards into tensor 
        reward = tf.convert_to_tensor(reward, dtype = tf.float32)
        
        # record operations for automatic differentiation.
        with tf.GradientTape() as tape:

            # obtain state value, V(s), and list of softmax probabilities for joint and action from current observation
            state_value, probs = self.A2C_SingleAction(state)
            
            # obtain state value prime, V(s'), from observation after a specified time step 
            state_value_prime, _ = self.A2C_SingleAction(state_prime)
            
            # ensure that state values obtained are scalar
            state_value = tf.squeeze(state_value)
            state_value_prime = tf.squeeze(state_value_prime)
            
            # obtain td error = reward(r) + discount_rate(gamma) * V(s') + V(s)
            td_error = reward + self.discount_rate * state_value_prime * (1 - is_done) - state_value
            
            # iterate over each pdf in probabilties list for each spiderbot leg
            for x in range(2):
                
                # create catergorical distribution from each pdf in probabilties list
                action_probs_list[x] = tfp.distributions.Categorical(probs = probs[x])
                
                # calculate log probablity of selected actions for a given state for each spiderbot leg
                log_prob_list[x] = action_probs_list[x].log_prob(self.actions[x])
            
                # calculate actor loss for each spiderbot leg 
                actor_loss_list[x] = -log_prob_list[x] * td_error
            
            # calculate critic loss = (td_error)^2
            critic_loss =  td_error**2
            
            # sum together all losses of actor critic combined hybrid model
            total_loss = critic_loss + sum(actor_loss_list)
        
        # computes actor critic gradient for all trainable variables using operations recorded in context of this tape
        gradients = tape.gradient(total_loss, self.A2C_SingleAction.trainable_variables)
        
        # apply actor critic gradients to all trainable variables in actor critic model
        self.A2C_SingleAction.optimizer.apply_gradients(zip(gradients, self.A2C_SingleAction.trainable_variables))
        
        # delete reference to tape
        del tape

        # return the total loss for logging
        return total_loss.numpy()[0]

    def save_all_models(self):
        
        """ save weights for all models """
        
        print("saving model!")
        
        # for two seperate actor and critic model 
        if self.model == "MAA2C":
            
            # save weights for each actor model
            for x in range(self.num_of_legs):

                self.MAA2C_actors_list[x].save_weights(self.MAA2C_actors_list[x].checkpoint_path)

            # save weights for critic
            self.MAA2C_Critic.save_weights(self.MAA2C_Critic.checkpoint_path)
        
        # for combined hybrid actor critic model v1
        elif self.model == "A2C_MultiAction":
            
            # save weights for actor critic
            self.A2C_MultiAction.save_weights(self.A2C_MultiAction.checkpoint_path)
        
        # for combined hybrid actor critic model v2
        elif self.model == "A2C_SingleAction":
            
            # save weights for actor critic
            self.A2C_SingleAction.save_weights(self.A2C_SingleAction.checkpoint_path)
            
        # for ddpg 
        elif self.model == "DDPG":
            
            # save weights for each actor, target_actor, critic, target_critic model
            self.DDPG_Actor.save_weights(self.DDPG_Actor.checkpoint_path)
            self.DDPG_Target_Actor.save_weights(self.DDPG_Target_Actor.checkpoint_path)
            self.DDPG_Critic.save_weights(self.DDPG_Critic.checkpoint_path)
            self.DDPG_Target_Critic.save_weights(self.DDPG_Target_Critic.checkpoint_path)
        
        # for mad3qn 
        elif self.model == "MAD3QN":
            
            # save weights for each q_eval model
            for x in range(self.num_of_legs):

                self.q_eval_list[x].save_weights(self.q_eval_list[x].checkpoint_path)
        
    def load_all_models(self):
        
        """ load weights for all models """
        
        print("loading model!")
        
        # for two seperate actor and critic model
        if self.model == "MAA2C":
            
            # save weights for each actor model
            for x in range(self.num_of_legs):
                
                self.MAA2C_actors_list[x].load_weights(self.MAA2C_actors_list[x].checkpoint_path).expect_partial()

            # save weights for critic
            self.MAA2C_Critic.load_weights(self.MAA2C_Critic.checkpoint_path).expect_partial()
        
        # for combined hybrid actor critic model
        elif self.model == "A2C_MultiAction":
            
            # load weights for actor critic
            self.A2C_MultiAction.load_weights(self.A2C_MultiAction.checkpoint_path).expect_partial()
        
        # for combined hybrid actor critic model v2
        elif self.model == "A2C_SingleAction":
            
            # load weights for actor critic
            self.A2C_SingleAction.load_weights(self.A2C_SingleAction.checkpoint_path).expect_partial()
 
        # for ddpg 
        elif self.model == "DDPG":
            
            # load weights for each actor, target_actor, critic, target_critic model
            self.DDPG_Actor.load_weights(self.DDPG_Actor.checkpoint_path).expect_partial()
            self.DDPG_Target_Actor.load_weights(self.DDPG_Target_Actor.checkpoint_path).expect_partial()
            self.DDPG_Critic.load_weights(self.DDPG_Critic.checkpoint_path).expect_partial()
            self.DDPG_Target_Critic.load_weights(self.DDPG_Target_Critic.checkpoint_path).expect_partial()

        # for mad3qn 
        elif self.model == "MAD3QN":
            
            # load weights for each q_eval model
            for x in range(self.num_of_legs):

                self.q_eval_list[x].load_weights(self.q_eval_list[x].checkpoint_path).expect_partial()