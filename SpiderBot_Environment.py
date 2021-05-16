#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# ME5406_Environment.py                                       #
# Author(s): Chong Yu Quan, Arijit Dasgupta                   #
# Email(s): chong.yuquan@u.nus.edu, arijit.dasgupta@u.nus.edu #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

""" 
The Environment Source Code for the ME5406 Project using the 3D Pybullet Physics Engine
Purpose 1 : "takes action" from agent by setting targets / reset()
Purpose 2 : returns rewards and observations to agent
"""

# Standard Imports
import pybullet as p
import pybullet_data
import numpy as np
import itertools
import copy


class p_gym:
    
    def __init__(self, connect_mode, num_of_legs, joint_velocities, joint_limit_lower, joint_limit_upper): 
       
        """ class constructor that initialises (connects) the PyBullet physics engine """
        
        # graphical user interface (GUI) client mode 
        if connect_mode == "GUI":

            self.client = p.connect(p.GUI)

        # direct link to physics engine
        elif connect_mode  == "DIRECT":

            self.client = p.connect(p.DIRECT)
        
        # adjust physics engine parameters if necessary (refer to documentation) --> not done for ME5406 Project
        # p.setPhysicsEngineParameter()
        
        # set gravity as -9.81 ms^-2
        p.setGravity(0, 0, -9.81)

        # add additional data path to access urdf files from pybullet
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # add plane to world
        self.plane_id = p.loadURDF("plane.urdf", basePosition = [0, 0, 0])
        
        # number of legs 
        self.num_of_legs = num_of_legs
        
        # add spider bot at world origin
        # Spider can have either 3, 4, 6 or 8 legs as we have the urdf for all these versions of the physical model
        # NOTE: All legs have 4 joints, which is hardcoded into the source code for this project
            
        self.spiderbot_id = p.loadURDF("Spiderbot_URDFs/SpiderBot_{}Legs/urdf/SpiderBot_{}Legs.urdf".format(self.num_of_legs, self.num_of_legs), basePosition = [0, 0, 0])
        
        # total number of joints
        self.num_of_joints = p.getNumJoints(self.spiderbot_id)
        
        # list of joints name indexed by their joint ids
        self.ord_joints_name = []
        
        # # prevent drifting
        # p.changeDynamics(self.spiderbot_id, -1, frictionAnchor = 1 )
        
        for x in range(self.num_of_joints):

            self.ord_joints_name.append(p.getJointInfo(self.spiderbot_id, x)[1].decode("utf-8"))
            
            # # prevent drifting
            # p.changeDynamics(self.spiderbot_id, x, frictionAnchor = 1)
        
        # array of joint limits indexed by [joint ids][lowerlimit(0) or upperlimit(1)] first initialised by zeros
        # necessary as alterations to joint limits not updated in getJointInfo in current PyBullet
        self.joint_limits = np.zeros((self.num_of_joints, 2))
        
        # apply joint limits to spiderbot
        for joint_id in range(self.num_of_joints):

            self.add_joint_limits(joint_id = joint_id, lower = joint_limit_lower[joint_id], 
                                  upper = joint_limit_upper[joint_id], mode = "first")
        
        # list of possible joint velocities
        self.joint_velocities = joint_velocities
        
        # list of permutations with repetition of joint velocites 
        self.joint_velocities_perm = [p for p in itertools.product(joint_velocities, repeat=4)]
        
        # disable velocity motor 
        p.setJointMotorControlArray(self.spiderbot_id, range(self.num_of_joints), p.VELOCITY_CONTROL, forces = [0 for x in range(self.num_of_joints)])
        
        # set all joints to be positional control
        # velocity and position constraint
        # error = position_gain*(desired_position-actual_position)+velocity_gain*(desired_velocity-actual_velocity)
        p.setJointMotorControlArray(self.spiderbot_id, range(self.num_of_joints), p.POSITION_CONTROL)
    
    def add_joint_limits(self, joint_id, lower, upper, mode):
        
        """ function to add jointLowerLimit, jointUpperLimit to joints (in radians for revolute joints) """
        
        # change joints limits
        p.changeDynamics(self.spiderbot_id, joint_id, jointLowerLimit = lower, jointUpperLimit = upper)
        
        if mode == "first":
        
            # update joint limits array
            self.joint_limits[joint_id][0] = lower
            self.joint_limits[joint_id][1] = upper    
    
    def return_obv_whole(self):
        
        """ function to return observation (flattened array) and reward (single value) for whole spiderbot """
        """ observation has length of 13 (base) + 15 * 32 (total number of joints/links (8 * 4)) = 493 for 8 legged spider """
        """ observations obtained from base, links and joint """
        """ base & link: position, orientation, velocities """
        """ joint: position, velocities """
             
        # obtain base position of 3 floats (x, y, z)
        base_position = np.array(p.getBasePositionAndOrientation(self.spiderbot_id)[0])
        
        # obtain base orientation of 4 floats (x, y, z, w)
        base_orientation = np.array(p.getBasePositionAndOrientation(self.spiderbot_id)[1])
        
        # obtain base linear velocity of 3 floats (v_x, v_y, v_z)
        base_lin_velocity = np.array(p.getBaseVelocity(self.spiderbot_id)[0])
        
        # obtain base angular velocity of 3 floats (w_x, w_y, w_z)
        base_ang_velocity = np.array(p.getBaseVelocity(self.spiderbot_id)[1])
        
        # concatenate and flatten base observations 
        observation = np.concatenate((base_position, base_orientation, base_lin_velocity, base_ang_velocity), axis = None)
        
        # obtain observations for each link of spiderbot
        link_data = np.array(p.getLinkStates(self.spiderbot_id, list(range(self.num_of_joints)), computeLinkVelocity = 1)) 
        
        # obtain observations for each joint of spiderbot
        joint_data = np.array(p.getJointStates(self.spiderbot_id, list(range(self.num_of_joints))))
        
        # iterate over all links/joint
        for joint_id in range(self.num_of_joints):
            
            # obtain position value of joint 
            joint_position = joint_data[joint_id][0]
            
            # obtain velocity value of joint
            joint_velocity = joint_data[joint_id][1]
            
            # obtain local position offset of inertial frame (center of mass) expressed in the URDF link frame
            # 3 floats (x, y, z)
            local_com_position = np.array(link_data[joint_id][2])
            
            # obtain local orientation (quaternion) offset of the inertial frame expressed in URDF link frame
            # 4 floats (x, y, z, w)
            local_orientation = np.array(link_data[joint_id][3])
            
            # obtain cartesian world linear velocity of 3 floats (v_x, v_y, v_z)
            local_lin_velocity = np.array(link_data[joint_id][6])
            
            # obtain cartesian world angular velocity of 3 floats (w_x, w_y, w_z)
            local_ang_velocity = np.array(link_data[joint_id][7])
            
            # concatenate and flatten observations 
            observation = np.concatenate((observation, joint_position, joint_velocity, local_com_position, 
                                          local_orientation, local_lin_velocity, local_ang_velocity), axis = None)
        
        return observation
        
    def return_obv_leg(self):
        
        """ function to return observation (flattened array) split by leg for MAA2C as actors/agents are decentralised"""
        """ observation has shape of (60, 8). 15 (num_observations) * 4 (number of joints/link) * 8 (number of legs) for 8 legged spider"""
        """ observations obtained from base, links and joint """
        """ base & link: position, orientation, velocities """
        """ joint: position, velocities """
        
        # list of legs         
        legs = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"][:self.num_of_legs]
        
        
        # iterate over legs
        for leg in legs:
            
            # set empty list to store joint ids for specific leg
            leg_joint_id = []
            
            # iterate over list of ordered joint names
            for joint in self.ord_joints_name:
                
                if leg in joint:
                    
                    # append joint ids related to specific leg
                    leg_joint_id.append(self.ord_joints_name.index(joint))
                    
            # obtain observations for each link of specific leg
            link_data = np.array(p.getLinkStates(self.spiderbot_id, leg_joint_id, computeLinkVelocity = 1)) 
            
            # obtain observations for each joint of specific leg
            joint_data = np.array(p.getJointStates(self.spiderbot_id, leg_joint_id))
            
            # iterate over relevant joint ids 
            for index in range(len(leg_joint_id)):
                
                # obtain position value of joint 
                joint_position = joint_data[index][0]

                # obtain velocity value of joint
                joint_velocity = joint_data[index][1]

                # obtain local position offset of inertial frame (center of mass) expressed in the URDF link frame
                # 3 floats (x, y, z)
                local_com_position = np.array(link_data[index][2])

                # obtain local orientation (quaternion) offset of the inertial frame expressed in URDF link frame
                # 4 floats (x, y, z, w)
                local_orientation = np.array(link_data[index][3])

                # obtain cartesian world linear velocity of 3 floats (v_x, v_y, v_z)
                local_lin_velocity = np.array(link_data[index][6])

                # obtain cartesian world angular velocity of 3 floats (w_x, w_y, w_z)
                local_ang_velocity = np.array(link_data[index][7])
                
                # concatenate and flatten first observations  
                if index == 0:
                    
                    leg_observation = np.concatenate((joint_position, joint_velocity, local_com_position, local_orientation,
                                                      local_lin_velocity, local_ang_velocity), axis = None)
                
                # concantenate and flatten subsequent observations to leg_observation
                else:
                    
                    leg_observation = np.concatenate((leg_observation, joint_position, joint_velocity, local_com_position, 
                                                      local_orientation, local_lin_velocity, local_ang_velocity), axis = None)
            
            # change shape from (len(leg_observation),) to (len(leg_observation), 1) for concatenation purposes
            leg_observation = leg_observation.reshape(len(leg_observation), 1)
            
            # copy first observation as consolidated observation array
            if legs.index(leg) == 0:

                observation = copy.deepcopy(leg_observation)

            # concatenate subsequent observations
            else: 

                observation = np.concatenate((observation, leg_observation), axis = 1)
        
        return observation
    
    def return_reward(self, terminate, time_step, target_location, forward_motion_reward = 500,
        forward_distance_reward = 250, sideways_velocity_punishment = 500, sideways_distance_penalty = 250, 
        time_step_penalty = 1, flipped_penalty = 500, goal_reward = 500, out_of_range_penalty = 500):
        
        """ function to return reward """
        """ reward based on velocity of COM of base in a specified direction (along x-axis) """
        """ reward punishes sideways (along y-axis) velocity """
        
        # reward based on velocity along x-axis
        # ensure that reward negative by subtracting a constant
        reward = forward_motion_reward * p.getBaseVelocity(self.spiderbot_id)[0][0] - forward_motion_reward
        
        # reward based on position from along x-axis
        # ensure that reward negative by subtracting the target location
        reward += forward_distance_reward * (p.getBasePositionAndOrientation(self.spiderbot_id)[0][0] - target_location)
        
        # punishes sideways (along y-axis) velocity                  
        reward -= sideways_velocity_punishment * abs(p.getBaseVelocity(self.spiderbot_id)[0][1])
        
        # punishes sideway (along y-axis) position
        reward -= sideways_distance_penalty * abs(p.getBasePositionAndOrientation(self.spiderbot_id)[0][1])
        
        # punishes time taken to reached goal / termination
        reward -=  time_step_penalty * time_step
        
#         # punishes spinning and tilting
#         quaternion = p.getBasePositionAndOrientation(self.spiderbot_id)[1]
#         roll = p.getEulerFromQuaternion(quaternion)[0]
#         pitch = p.getEulerFromQuaternion(quaternion)[1]
#         yaw = p.getEulerFromQuaternion(quaternion)[2]
#         reward -= 1000 * (roll**2 + pitch**2 + yaw**2)
        
        # if spider is "flipped"
        if terminate == 1:
            
            # decrease reward
            reward -= flipped_penalty
        
        # if spider reaches finishing line
        elif terminate == 2:
            
            # increase reward 
            
            reward += goal_reward
            
        # if spider goes out of range
        elif terminate == 3 or terminate == 4 :
            
            # increase reward 
            
            reward -= out_of_range_penalty
        
        return reward
        
    def set_target_whole(self, model, actions):

        """ function to set target velocities for all joints """  
        """ takes in list of actions with shape (8, 1) if model = "MAA2C" or "A2C_MultiAction" or "DDPG" """
        
        if model == "MAA2C" or model == "A2C_MultiAction" or model == "MAD3QN":
        
            # iterate over each leg of spiderbot
            for leg in range(self.num_of_legs):

                # select target positions from permuted list 
                target_velocities = self.joint_velocities_perm[actions[leg]]

                # set target position for joints of the leg 
                p.setJointMotorControlArray(self.spiderbot_id, list(range(4 * leg, 4 * (leg + 1), 1)), p.POSITION_CONTROL, 
                                            targetVelocities = target_velocities)
        
        elif model == "DDPG":
            
            # iterate over each joint of spiderbot
            for joint_id in range(self.num_of_joints):
                
                # set target position for joint 
                p.setJointMotorControl2(self.spiderbot_id, joint_id, p.POSITION_CONTROL, 
                                        targetVelocity = actions[joint_id])
    
    def set_target_joint(self, action):
        
        """ function to set target velocites for specific leg for the A2C_SingleAction algorithm"""  
        """ action list of shape (2, ). 0th index is selected joint(joint_index). 1st index is selected action """
        
        # set target position for joint 
        p.setJointMotorControl2(self.spiderbot_id, action[0], p.POSITION_CONTROL, 
                                targetVelocity = self.joint_velocities[action[1]])
        
    def is_terminate(self, start_time, end_time, target_location):
        
        """ function to check if episode should be terminated """
        """ termination condition 1 : spiderbot is "flipped" """
        """ termination condition 2 : spiderbot reaches a target location along the x-axis """
        """ termination condition 3 : episode timeout (real-time) """
        
        # contact points of spiderbot with plane
        # list should have a length of self.num_of_legs (for each end link per leg) if spider is walking normally 
        contact_points_tuple = p.getContactPoints(bodyA = self.spiderbot_id, bodyB = self.plane_id)
        
        # list of joint_ids of end links of for each leg
        end_joint_ids = [(4 * x + 3) for x in range(self.num_of_legs)]
        
        # list to store contact points that are not end links
        end_contact_points_list = []
        
        # iterate over contact list
        for x in range(len(contact_points_tuple)):
            
            # append to end contact points list if joint id from an end link
            if contact_points_tuple[x][3] not in end_joint_ids:
                
                end_contact_points_list.append(contact_points_tuple[x])
        
        # checks if any links (other than end links) of the spiderbot touches plane, i.e "flipped" 
        if len(end_contact_points_list) != 0:
            
            return 1
        
        # checks if base of object crosses a specified finishing line along x-axis 
        elif p.getBasePositionAndOrientation(self.spiderbot_id)[0][0] >= target_location:
            
            return 2
        
        # checks if base of object crosses goes beyond a specified negative location on x-axis 
        elif p.getBasePositionAndOrientation(self.spiderbot_id)[0][0] <= -1:
            
            return 3
        
        # checks if base of object crosses goes beyond a specified range from the y-axis 
        elif abs(p.getBasePositionAndOrientation(self.spiderbot_id)[0][1]) >= 1:
            
            return 4
        
        # check if episode is longer than a specified episode timeout
        elif (end_time - start_time) >= (60 * 10):
            
            return 5
        
        # else return 0
        else:
            
            return 0

    def reset(self):
        
        """ function to reset the environment """
        
        # resets to a clean slate
        p.resetSimulation()
        
        # set gravity as -9.81 ms^-2
        p.setGravity(0, 0, -9.81)

        # add additional data path to access urdf files from pybullet
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # add plane to world
        self.plane_id = p.loadURDF("plane.urdf", basePosition = [0, 0, 0])

        # add spider bot at world origin 
        self.spiderbot_id = p.loadURDF("Spiderbot_URDFs/SpiderBot_{num_legs}Legs/urdf/SpiderBot_{num_legs}Legs.urdf".format(num_legs = self.num_of_legs), basePosition = [0, 0, 0])

            
        # reapply joint limits to all the joints 
        for joint_id in range(self.num_of_joints):
        
            self.add_joint_limits(joint_id = joint_id, lower = self.joint_limits[joint_id][0] , 
                                  upper = self.joint_limits[joint_id][1], mode = False)
        
        # disable velocity motor 
        p.setJointMotorControlArray(self.spiderbot_id, range(self.num_of_joints), p.VELOCITY_CONTROL, forces = [0 for x in range(self.num_of_joints)])
        
        # reapply all joints to be positional control
        p.setJointMotorControlArray(self.spiderbot_id, range(self.num_of_joints), p.POSITION_CONTROL)