#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# SpiderBot_Walk.py                                              #
# Author(s): Chong Yu Quan, Arijit Dasgupta                   #
# Email(s): chong.yuquan@u.nus.edu, arijit.dasgupta@u.nus.edu #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

""" This script has the main walk function that trains the spiderbot """

# Standard Imports
import pybullet as p
import pandas as pd
import numpy as np
import math
import time
import os
# SpiderBot Imports
from SpiderBot_Agent import Agent
from SpiderBot_Environment import p_gym

def walk(mode, training_name, time_step_size, model, save_best_model ,save_data, num_of_legs, episodes, joint_limit_lower, 
        joint_limit_upper, joint_velocities, discount_rate, lr_actor, lr_critic, target_location, tau, max_mem_size, batch_size, 
        noise, max_action, min_action, epsilon, epsilon_decay, epsilon_min, update_target, forward_motion_reward, forward_distance_reward, 
        connect_mode, sideways_velocity_punishment, sideways_distance_penalty, time_step_penalty, flipped_penalty, goal_reward, out_of_range_penalty):
    
    #~~~~~~~~~~~~~~~~ INIITALISE Environment & Agent ~~~~~~~~~~~~~~~~~~~~#

    # create p_gym instanance
    gym = p_gym(connect_mode = connect_mode, num_of_legs = num_of_legs, joint_velocities = joint_velocities, 
                joint_limit_lower = joint_limit_lower, joint_limit_upper = joint_limit_upper)

    # create agent instance
    if model == "MAA2C" or model == "A2C_MultiAction" or model == "MAD3QN":
        
        # Considers the permutation of actions in action space for a leg as action commands are given to an entire leg at once
        agent = Agent(model = model, num_of_legs = gym.num_of_legs, num_of_joints = gym.num_of_joints, 
                      discount_rate = discount_rate, lr_actor = lr_actor, lr_critic = lr_critic, 
                      action_space = len(gym.joint_velocities_perm), tau = tau, max_mem_size = max_mem_size, 
                      batch_size = batch_size, noise = noise, max_action = max_action, min_action = min_action, 
                      epsilon = epsilon, epsilon_decay = epsilon_decay, epsilon_min = epsilon_min, 
                      update_target = update_target, training_name = training_name)
    
    elif model == "A2C_SingleAction" or model == "DDPG":
        # only considers the action space for a joint as the action commands are given to individual joints
        agent = Agent(model = model, num_of_legs = gym.num_of_legs, num_of_joints = gym.num_of_joints, 
                      discount_rate = discount_rate, lr_actor = lr_actor, lr_critic = lr_critic, 
                      action_space = len(gym.joint_velocities), tau = tau, max_mem_size = max_mem_size, 
                      batch_size = batch_size, noise = noise, max_action = max_action, min_action = min_action,
                      epsilon = epsilon, epsilon_decay = epsilon_decay, epsilon_min = epsilon_min, 
                      update_target = update_target, training_name = training_name)
    else:
        raise ValueError("Invalid model/algorithm set in config")

    #~~~~~~~~~~~~~~~~ INITIALISE LOGS ~~~~~~~~~~~~~~~~~~~~#

    # average velocity variable to track progress for model saving
    best_avg_vel = -math.inf
    
    # list of furthest distance variable travelled
    dist_log = []
    
    # list to store average velocity for each episode
    avg_vel_log = []

    # list to store time taken for episode to finish
    time_taken_log = []

    # list to store success (terminate == 2) for each episode
    success_log = []

    # list to store whenever the SpiderBot falls (terminate == 1) for each episode
    fall_log = []

    # list to store whenever the SpiderBot goes too far back (terminate == 3) for each episode
    backward_log = []

    # list to store whenever the SpiderBot goes too far sideways (terminate == 4) for each episode
    sideways_log = []

    # list to store whenever SpiderBot takes too long during the episode  (terminate == 5) for each episode
    time_limit_log = []

    # list to store training loss for all neural networks combined for each model/algorithm
    nn_training_loss_log = []
    nn_training_episode_log = []

    #~~~~~~~~~~~~~~~~ LOAD MODELS (if necessary) and start training ~~~~~~~~~~~~~~~~~~~~#

    # load saved models if testing
    if mode == "test" or mode == "load_n_train":

        agent.load_all_models()

    # iterate over specified number of episodes
    for x in range(episodes):
        
        # allow 3 seconds the robot to drop and land on the plane
        # default time step is 1/240 seconds, hence 3 / (1/240) = 720
        for i in range(720):
            
            p.stepSimulation()
        
        # record start time of episode
        start_time = time.time()
        
        # generate first observations of the env for actor and critic for seperate actor and critic
        if agent.model == "MAA2C":
        
            # decentralised actors/agents, where one leg is an agent --> only focused on its own leg
            actor_observations = gym.return_obv_leg()
            # centralised critic --> focused on entire bot
            critic_observations = gym.return_obv_whole()
        
        # generate first observations of the env for actor and critic for combined hybrid networks
        elif agent.model == "A2C_MultiAction" or agent.model == "A2C_SingleAction" or agent.model == "DDPG" or agent.model == "MAD3QN":   

            observations = gym.return_obv_whole()

        # list to track velocity over episode
        vel = []
        
        # furthest distance travelled 
        best_dist = -math.inf
        
        # boolean to show if episode is terminated
        is_done = 0
        
        # time step
        time_step = 0

        #~~~~~~~~~~~~~~~~ START OF EPISODE ~~~~~~~~~~~~~~~~~~~~#

        # run episode till terminates
        while is_done == 0:


            #~~~~~~~~~~~~~~~~ SELECT ACTIONS ~~~~~~~~~~~~~~~~~~~~#
            
            # actor models to generate actions for seperate actor and critic
            if agent.model == "MAA2C":
                
                actions = agent.select_actions(actor_observations, mode = mode)
                
            # combined hybrid actor critic model to generate actions
            elif agent.model == "A2C_MultiAction" or agent.model == "A2C_SingleAction" or agent.model == "DDPG" or agent.model == "MAD3QN":   
                
                actions = agent.select_actions(observations, mode = mode)
            #~~~~~~~~~~~~~~~~ EXECUTE ACTIONS ~~~~~~~~~~~~~~~~~~~~#

            # carrying out MULTIPLE actions in the SAME TIME STEP
            if agent.model == "MAA2C" or agent.model == "A2C_MultiAction" or agent.model == "DDPG" or agent.model == "MAD3QN":
                
                # gym to set the target joint positions
                gym.set_target_whole(model, actions)
                
            # carrying out only a SINGLE action per TIME STEP
            elif agent.model == "A2C_SingleAction":
                
                # gym to set the target joint positions
                gym.set_target_joint(actions)
            # allow spiderbot to actuate actions based on user-defined time step size (in seconds)
            for i in range(max(int(time_step_size * 240),1)):
                
                # rum simulation for default time step of 1/240 seconds
                p.stepSimulation()
                
                # add time step
                time_step += 1./240
            # record end time
            end_time = time.time()

            #~~~~~~~~~~~~~~~~ RECORD STATES AFTER ACTION EXECUTED ~~~~~~~~~~~~~~~~~~~~#

            if agent.model == "MAA2C":
                
                # generate observations for actor model after specified time step
                actor_observations_prime = gym.return_obv_leg()

                # generate observations for critic model after specified time step
                critic_observations_prime = gym.return_obv_whole()
            
            elif agent.model == "A2C_MultiAction" or agent.model == "A2C_SingleAction" or agent.model == "DDPG" or agent.model == "MAD3QN":   
                
                # generate observations for combined hybrid actor-critic models after specified time step
                observations_prime = gym.return_obv_whole()


            #~~~~~~~~~~~~~~~~ CHECK TERMINATION, REWARD, vel, DIST, BEST_DIST & Update Replay Memory ~~~~~~~~~~~~~~~~~~~~#

            # check if episode fullfills any of the terminating conditions
            terminate = gym.is_terminate(start_time = start_time, end_time = end_time, target_location = target_location)

            # terminate episode if termination conditions are met
            if terminate != 0:
                
                is_done = 1

            if (mode == "train" or mode == "load_n_train"):
                # return reward after specified time step
                reward = gym.return_reward(terminate, time_step, target_location, forward_motion_reward,
                forward_distance_reward, sideways_velocity_punishment, sideways_distance_penalty, 
                time_step_penalty, flipped_penalty, goal_reward, out_of_range_penalty)
                
                # store memory for state, action, state_prime, reward, terminal flag for ddpg
                if agent.model == "DDPG" or agent.model == "MAD3QN":

                    agent.store_memory(observations, actions, reward, observations_prime, is_done)
            
            # add reward to velocity list
            vel.append(p.getBaseVelocity(gym.spiderbot_id)[0][0])
            
            # distance travelled 
            dist = p.getBasePositionAndOrientation(gym.spiderbot_id)[0][0]
            
            # store in best_distance if larger
            if dist >= best_dist:
                
                best_dist = dist

            #~~~~~~~~~~~~~~~~ APPLY GRADIENTS (i.e. backpropagation) ~~~~~~~~~~~~~~~~~~~~#

            # apply gradients for learning during training and log the NN loss
            if mode == "train" or mode == "load_n_train":
                
                # apply gradients for seperate actor and critic
                if agent.model == "MAA2C":
                    
                    nn_training_loss_log.append(agent.apply_gradients_MAA2C(critic_observations, reward, critic_observations_prime, is_done, actor_observations))
                    nn_training_episode_log.append(x+1)

                # apply gradients for combined hybrid actor critic 
                elif agent.model == "A2C_MultiAction": 
                    
                    nn_training_loss_log.append(agent.apply_gradients_A2C_MultiAction(observations, reward, observations_prime, is_done))
                    nn_training_episode_log.append(x+1)

                    # apply gradients for combined hybrid actor critic 
                elif agent.model == "A2C_SingleAction": 
                    
                    nn_training_loss_log.append(agent.apply_gradients_A2C_SingleAction(observations, reward, observations_prime, is_done))
                    nn_training_episode_log.append(x+1)

                # apply gradients for ddpg from sample memory
                elif agent.model == "DDPG":
                    
                    nn_training_loss_log.append(agent.apply_gradients_DDPG())
                    nn_training_episode_log.append(x+1)

                # apply gradients for d3qn from sample memory
                elif agent.model == "MAD3QN":
                    
                    nn_training_loss_log.append(agent.apply_gradients_MAD3QN())
                    nn_training_episode_log.append(x+1)

            # update observations variables of previous time step for the each type of models
            # separate networks for actor and critic
            if agent.model == "MAA2C":    
                
                actor_observations = actor_observations_prime
                critic_observations = critic_observations_prime

            # hybrid combined networks for actor and critic
            elif agent.model == "A2C_MultiAction" or agent.model == "A2C_SingleAction" or agent.model == "DDPG" or agent.model == "MAD3QN":   
                
                observations = observations_prime

        #~~~~~~~~~~~~~~~~ UPDATE ALL LOGS ~~~~~~~~~~~~~~~~~~~~#

        # calculate average velocity throughout episode
        avg_vel = np.mean(vel)
        
        # append vel to vel log at end of episode
        avg_vel_log.append(avg_vel)

        # append best dist to dist_log at end of episode
        dist_log.append(best_dist)

        # append time taken for episode to terminate
        time_taken_log.append(time.time() - start_time)
        
        # if spiderbot crosses finishing line, consider episode a success
        if terminate == 2:
            method_of_termination = "Reached Goal"
            success_log.append(1)
        # else its a fail
        else:
            success_log.append(0)

        if terminate == 1:
            method_of_termination = "Fell Down"
            fall_log.append(1)
        else:
            fall_log.append(0)

        if terminate == 3:
            method_of_termination = "Went Too Far Back"
            backward_log.append(1)
        else:
            backward_log.append(0)

        if terminate == 4:
            method_of_termination = "Went Too Far Sideways"
            sideways_log.append(1)
        else:
            sideways_log.append(0)

        if terminate == 5:
            method_of_termination = "Time Limit Exceeded"
            time_limit_log.append(1)
        else:
            time_limit_log.append(0)

        # if average velocity has improved to be better than best vel (initialised to 0)
        if avg_vel >= best_avg_vel and terminate == 2:
            
            # average vel is new best vel
            best_avg_vel = avg_vel
            
            # store the model responsible for this vel gain
            if (mode == "train" or mode == "load_n_train") and save_best_model == True:
                
                agent.save_all_models()
        
        # log episode performance
        print(f"episode: {x}, avg_vel: {avg_vel}, best_dist: {best_dist}, best_avg_vel: {best_avg_vel}, " + \
         f"method of termination: {method_of_termination}, success: {success_log[x]}, time taken: {time_taken_log[x]:.1f}s")\

        if save_data:

            # combine logs into a pandas dataframe and save it to a CSV every episode (file should be overwritten every episode)

            training_df = pd.DataFrame(list(zip(list(range(1,x+1)),avg_vel_log, dist_log, success_log, fall_log, backward_log,
            sideways_log, time_limit_log, time_taken_log)),
            columns = ['episodes', 'avg_vel', 'dist', 'success', 'fall', 'backward', 'sideways', 'time_limit', 'time_taken'])
            nn_training_loss_df = pd.DataFrame(list(zip(nn_training_episode_log,  nn_training_loss_log)), columns = ['episode', 'nn_training_loss'])

            training_df.to_csv("Training_Logs/{}_logs.csv".format(training_name), index = False)
            nn_training_loss_df.to_csv("Training_Logs/{}_NN_loss.csv".format(training_name), index = False)

        # reset the environemnt for the next episode
        gym.reset()
    
    # disconnect from physics engine after training/testing
    p.disconnect()

    return