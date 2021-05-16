#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# SpiderBot_Neural_Network.py                                    #
# Author(s): Chong Yu Quan, Arijit Dasgupta                   #
# Email(s): chong.yuquan@u.nus.edu, arijit.dasgupta@u.nus.edu #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

""" 
Neural Network model classes using TensorFlow 2.3.1
Purpose : efficiently generate model architecture by building from subclasses
5 Algorithm networks available: MAD3QN, MAA2C, A2CMA, A2CSA, DDPG 
"""

# Standard Import
import tensorflow as tf

class fc_block(tf.keras.layers.Layer):
    
    def __init__(self, h_units, weight_decay, dropout_rate):
        
        """ class constructor that creates the layers attributes for fc_block """
        
        # inherit class constructor attributes from tf.keras.layers.Layer
        super(fc_block, self).__init__()
        
        # add dense layer attribute with L2 Regulariser
        self.dense = tf.keras.layers.Dense(h_units, use_bias = False, kernel_regularizer = 
                                            tf.keras.regularizers.l2(l = weight_decay))
        
        # add batch norm layer attribute
        self.batch_norm = tf.keras.layers.BatchNormalization()
    
    def call(self, inputs, training = False):
        
        """ function for forward pass of model """
        """ includes training argument as batch_norm functions differently during training and during inference """
        
        # inputs --> dense --> relu --> batch_norm --> output
        x = self.dense(inputs)
        x = tf.nn.relu(x)
        x = self.batch_norm(x, training = training)

        return x
    
class fc_model(tf.keras.Model):
    
    def __init__(self, model, num_of_legs, num_of_joints, h_units, weight_decay, dropout_rate, num_of_outputs, training_name):
        
        """ class constructor that creates the layers attributes for fully connected model """
        
        # inherit class constructor attributes from tf.keras.Model
        super(fc_model, self).__init__()
        
        # model name
        self.model_name = None
        
        # type of model architecture
        self.model = model
        
        # number of legs 
        self.num_of_legs = num_of_legs
        
        # number of joints
        self.num_of_joints = num_of_joints
        
        # checkpoint directory
        self.checkpoint_dir = "Saved_Models/" + training_name + "_" + "best_models/"
        
        # checkpoint filepath 
        self.checkpoint_path = None
        
        # create intended number of dqn_block attributes
        self.block_1 = fc_block(h_units[0], weight_decay[0], dropout_rate[0])
        self.block_2 = fc_block(h_units[1], weight_decay[1], dropout_rate[1])
        self.block_3 = fc_block(h_units[2], weight_decay[2], dropout_rate[2])
            
        # create final output layer attribute
        if self.model == "MAA2C_Actor":
            
            # output layer is probability density function for actions in discretised action space
            self.outputs = tf.keras.layers.Dense(num_of_outputs, activation = 'softmax')
        
        elif self.model == "MAA2C_Critic": 
            
            # output layer is state value, V, for a given state
            self.outputs = tf.keras.layers.Dense(num_of_outputs)
            
        elif self.model == "A2C_MultiAction":
            
            # combined actor critic model which takes multiple actions per time step
            # gives state value as well as output probability density function for each leg
            self.outputs_V = tf.keras.layers.Dense(1)
            self.outputs_1 = tf.keras.layers.Dense(num_of_outputs, activation = 'softmax')
            self.outputs_2 = tf.keras.layers.Dense(num_of_outputs, activation = 'softmax')
            self.outputs_3 = tf.keras.layers.Dense(num_of_outputs, activation = 'softmax')

            if self.num_of_legs >= 4:

                self.outputs_4 = tf.keras.layers.Dense(num_of_outputs, activation = 'softmax')

            if self.num_of_legs >= 6:

                self.outputs_5 = tf.keras.layers.Dense(num_of_outputs, activation = 'softmax')
                self.outputs_6 = tf.keras.layers.Dense(num_of_outputs, activation = 'softmax')
            
            if self.num_of_legs == 8:
                
                self.outputs_7 = tf.keras.layers.Dense(num_of_outputs, activation = 'softmax')
                self.outputs_8 = tf.keras.layers.Dense(num_of_outputs, activation = 'softmax')
        
        elif self.model == "A2C_SingleAction":
        
            # combined actor critic model that takes single action per time step
            # gives state value, probability density function for each joint, pdf for range of actions
            self.outputs_V = tf.keras.layers.Dense(1)
            self.outputs_num_of_joints = tf.keras.layers.Dense(self.num_of_joints, activation = 'softmax')
            self.outputs_actions = tf.keras.layers.Dense(num_of_outputs, activation = 'softmax')
        
        elif self.model == "DDPG_Actor":
            
            # output layer with continuous action for each joint
            self.outputs = tf.keras.layers.Dense(num_of_outputs, activation = 'tanh')
        
        elif self.model == "DDPG_Critic": 

            # output layer is state-action value, Q, for a given state and action
            self.outputs = tf.keras.layers.Dense(num_of_outputs)
            
        elif self.model == "MAD3QN":
            
            self.outputs_V = tf.keras.layers.Dense(1) 
            self.outputs_A = tf.keras.layers.Dense(num_of_outputs)
            
    def call(self, inputs, training = False):
        
        """ function for forward pass of model """
        """ includes training argument as batch_norm functions differently during training and during inference """
            
        # input --> block_1 --> block_2 --> block_3 
        x = self.block_1(inputs, training = training)
        x = self.block_2(x, training = training)
        x = self.block_3(x, training = training)

        # outputs for single-agent hybrid A2C algorithm with a decentralised policy for each leg
        if self.model == "A2C_MultiAction":

            # block_3 --> (V + pdf for each leg)
            v = self.outputs_V(x) 
            pol_1 = self.outputs_1(x) 
            pol_2 = self.outputs_2(x)
            pol_3 = self.outputs_3(x)

            # create list of probabilities
            pol = [pol_1, pol_2, pol_3]

            if self.num_of_legs >= 4:

                pol_4 = self.outputs_4(x)
                pol.append(pol_4) 

            if self.num_of_legs >= 6:

                pol_5 = self.outputs_5(x)
                pol_6 = self.outputs_6(x)
                pol.append(pol_5)
                pol.append(pol_6)

            if self.num_of_legs >= 8:

                pol_7 = self.outputs_7(x) 
                pol_8 = self.outputs_8(x)
                pol.append(pol_7)
                pol.append(pol_8)

            return v, pol

        # outputs for single-agent hybrid A2C algorithm with a single centralised policy
        elif self.model == "A2C_SingleAction":

            # block_3 --> V
            v = self.outputs_V(x) 

            # block_3 --> pdf to select a joint
            pol_joint = self.outputs_num_of_joints(x)

            # block_3 --> pdf to select an action for selected joint
            pol_action = self.outputs_actions(x)

            pol = [pol_joint, pol_action]

            return v, pol

        # outputs for multi-agent dueling_double_dqn 
        elif self.model == "MAD3QN":
            
            # block_3 --> V
            V = self.outputs_V(x)
            
            # block_3 --> A (advantange for each action)
            A = self.outputs_A(x)
            
            # Q = V + A - mean(A)
            Q = (V + A - tf.math.reduce_mean(A, axis = 1, keepdims = True))
            
            return Q
            
        # output for MAA2C or DDPG
        else:    

            # block_3 --> outputs 
            x = self.outputs(x)

        return x