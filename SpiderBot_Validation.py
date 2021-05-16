#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# SpiderBot_Validation.py                                        #
# Author(s): Chong Yu Quan, Arijit Dasgupta                   #
# Email(s): chong.yuquan@u.nus.edu, arijit.dasgupta@u.nus.edu #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

"""
In this script, you can test & observe the performance of a model in validation-mode.
This means that the spiderbot is not training, but it is merely taking actions depending on the state
To run the validation of a model --> specify the training name and the algorithm used below
NOTE: The parameters in the walk function have been hardcoded for the SpiderBot Project Validation by Professors/TAs
"""

# Standard Import
import math
# SpiderBot Imports
from SpiderBot_Walk import walk

#~~~~~~~~~~~~ VALIDATION CONFIG SETUP ~~~~~~~~~~~~#
# specify the training name
training_name = "DDPG_FullyTrained" # DDPG_FullyTrained, DDPG_PartiallyTrained
# Specify the Algorithm used
model = "DDPG"
# What is the target location you wish to set (in metres)? DDPG_FullyTrained can go forward up to 9 metres
target_location = 8
# How many episodes do you wish to visualise the SpiderBot for? 
# Note that every episode will be exactly the same as there is no training going on
episodes = 100000000000 # A large number is set to put the simulation on loop
#~~~~~~~~~~~~ END OF CONFIG ~~~~~~~~~~~~~~#


# set discrete joint velocities for non-DDPG algorithms
if model == "A2C_SingleAction":
       joint_velocities = [x * 0.5 for x in range(-20, 21, 1)]
else:
       joint_velocities = [-10, -5, 0, 5, 10]

# set upper and lower joint limits
joint_limit_lower = [math.radians(-60) for x in range(32)]
joint_limit_upper = [math.radians(60) for x in range(32)]

# Run the walk function in test mode to observe a trained SpiderBot
walk(
mode = "test", 
model = model,
connect_mode = "GUI",
training_name = training_name, 
save_best_model = False,
save_data = False, 
num_of_legs = 8, 
episodes = episodes,
time_step_size = 120./240, 
joint_limit_lower = joint_limit_lower, 
joint_limit_upper = joint_limit_upper,
joint_velocities = joint_velocities,
discount_rate = None,
lr_actor = None,
lr_critic = None, 
target_location = target_location, 
tau = None, 
max_mem_size = 1000000,
batch_size = None,
noise = None,
max_action = 10, 
min_action = -10, 
epsilon = None, 
epsilon_decay = None, 
epsilon_min = None,
update_target = None,
forward_motion_reward = None,
forward_distance_reward = None, 
sideways_velocity_punishment = None,
sideways_distance_penalty = None,
time_step_penalty = None,
flipped_penalty = None,
goal_reward = None,
out_of_range_penalty = None
)

