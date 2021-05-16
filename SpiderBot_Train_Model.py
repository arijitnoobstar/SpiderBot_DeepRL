#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# SpiderBot_Train_Model.py 									  #
# Author(s): Chong Yu Quan, Arijit Dasgupta 				  #
# Email(s): chong.yuquan@u.nus.edu, arijit.dasgupta@u.nus.edu #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

"""
This script is responsible for the training for the spiderbot RL agent.
User must specify parameters for training in the 3 CONFIG sections below.
"""

# Standard Import
import math
# SpiderBot Imports
from SpiderBot_Walk import walk
from SpiderBot_Postprocessing import post_process

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONFIGURATION FOR TRAINING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

#~~~~~~~~~~~~~~~~~~~~~~~~~~~ GENERAL CONFIG ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# SPECIFY TRAINING NAME
training_name = "insert_training_name_here"
# Specify the Algorithm/Model to use
model = "DDPG" # 'MAA2C', 'MAD3QN' --> multi-agent approch OR 'A2C_MultiAction', 'A2C_SingleAction', 'DDPG' --> single-agent approach
# Specify number of legs for SpiderBot to have (we recognise that a bot with less than 8 legs is not a spider anymore)
num_of_legs = 8 # 3, 4, 6, 8
# Specify number of Episodes to run for
episodes = 1
# Set the target location for the SpiderBot to walk to
target_location = 3
# GUI mode of training or train withour GUI (faster)
use_GUI = True
# Do you want to conduct postprocessing after training?
do_post_process = True
# Do you want to save the best versions of the model?
save_best_model = True
# Do you want to save the CSV data from training?
save_data = True
#~~~~~~~~~~~~~~~~~~~~~~~~~~~ HYPERPARAMETER CONFIG ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# set the time length (in seconds) for each time step to split the continuous time space into discrete steps
time_step_size = 120./240
# Upper and Lower Joint limits in angle in degrees
upper_angle = 60
lower_angle = -60
# Specify learning rates for actors and critics, for MAD3QN, only the actor learning rate is used
lr_actor = 0.00005
lr_critic = 0.0001
# Specify the discount rate
discount_rate = 0.9
# number of time_steps to do a hard copy for target networks in MAD3QN and DDPG 
# Otherwise enter None for Softcopy
update_target = None
# Tau value for soft copy of online network weights to target networks for MAD3QN and DDPG
tau = 0.005
# Replay memory size and batch size for MAD3QN and DDPG
max_mem_size = 1000000
batch_size = 512
# Continuous action space range in units of rad/s of joints and noise stddev for DDPG.
max_action = 10
min_action = -10
noise = 1
# epsilon value for starting, minimum and decay for MAD3QN (as it is a value approximator)
epsilon = 1
epsilon_decay = 0.0001
epsilon_min = 0.01

#~~~~~~~~~~~~~~~~~~~~~~~~~~~ REWARD STRUCTURE CONFIG ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# reward for forward velocity in the correct direction
forward_motion_reward = 500
# reward for being close to the target location
forward_distance_reward = 250
# penalty for having sideways velocity
sideways_velocity_punishment = 500
# penalty for moving sideways from axis of walk
sideways_distance_penalty = 250
# penalty applied multiplied to the total number of seconds passed during training
time_step_penalty = 1
# penalty for terminating by flipping
flipped_penalty = 500
# reward for reaching the Target
goal_reward = 500
# penalty for terminating by going out of range (more than 1m sideways and backward from origin)
out_of_range_penalty = 500

############################## END OF CONFIGURATION #################################


# set discrete joint velocities for non-DDPG algorithms
if model == "A2C_SingleAction":
       joint_velocities = [x * 0.5 for x in range(-20, 21, 1)]
else:
       joint_velocities = [-10, -5, 0, 5, 10]

# set upper and lower joint limits
joint_limit_lower = [math.radians(lower_angle) for x in range(4 * num_of_legs)]
joint_limit_upper = [math.radians(upper_angle) for x in range(4 * num_of_legs)]

# set pybullet connect mode
if use_GUI:
	connect_mode = "GUI"
else:
	connect_mode = "DIRECT"

# Run the walk function in train mode to train the SpiderBot RL agent
walk(
mode = "train", 
model = model,
connect_mode = connect_mode,
training_name = training_name, 
save_best_model = save_best_model,
save_data = save_data, 
num_of_legs = num_of_legs, 
episodes = episodes,
time_step_size = time_step_size, 
joint_limit_lower = joint_limit_lower, 
joint_limit_upper = joint_limit_upper,
joint_velocities = joint_velocities,
discount_rate = discount_rate,
lr_actor = lr_actor,
lr_critic = lr_critic, 
target_location = target_location, 
tau = tau, 
max_mem_size = max_mem_size,
batch_size = batch_size,
noise = noise,
max_action = max_action, 
min_action = min_action, 
epsilon = epsilon, 
epsilon_decay = epsilon_decay, 
epsilon_min = epsilon_min,
update_target = update_target,
forward_motion_reward = forward_motion_reward,
forward_distance_reward = forward_distance_reward, 
sideways_velocity_punishment = sideways_velocity_punishment,
sideways_distance_penalty = sideways_distance_penalty,
time_step_penalty = time_step_penalty,
flipped_penalty = flipped_penalty,
goal_reward = goal_reward,
out_of_range_penalty = out_of_range_penalty
)

# if specified, conduct post_processing
if do_post_process:
	post_process(training_name)