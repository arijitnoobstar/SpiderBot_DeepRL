#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# SpiderBot_Postprocessing.py 									  #
# Author(s): Chong Yu Quan, Arijit Dasgupta 				  #
# Email(s): chong.yuquan@u.nus.edu, arijit.dasgupta@u.nus.edu #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

"""
This script is responsible for the postprocessing of results from the training of the spiderbot
User must either specify training name for post processing manually in this script, or call
the post_process function from SpiderBot_Walk.py
"""

# Standard Imports
import os
import shutil
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

# SPECIFY TRAINING NAME FOR POST PROCESSING FOR MANUAL USE OF THIS SCRIPT
training_name = "insert_training_name_here"
def post_process(training_name):

	# extract csv files into Pandas DataFrames
	training_logs = pd.read_csv("Training_Logs/{}_logs.csv".format(training_name), header = 0)
	nn_loss = pd.read_csv("Training_Logs/{}_NN_loss.csv".format(training_name), header = 0)

	# make directory for plots, if is already exists, override it
	try:
		os.mkdir("Training_Plots/" + training_name)
	except:
		shutil.rmtree("Training_Plots/" + training_name)
		os.mkdir("Training_Plots/" + training_name)

	plt.title("Average Velocity vs Number of episodes")
	plot_1 = sns.lineplot(data = np.array(training_logs['avg_vel']))
	plt.ylabel("Average Velocity")
	plt.xlabel("Number of episodes")
	plt.savefig("Training_Plots/" + training_name  + "/" + training_name + "_avg_vel_vs_episodes.pdf", 
				bbox_inches = 'tight')
	plt.close()

	plt.title("Furthest Distance Travelled vs Number of episodes")
	plot_2 = sns.lineplot(data = np.array(training_logs['dist']))
	plt.ylabel("Distance")
	plt.xlabel("Number of episodes")
	plt.savefig("Training_Plots/" + training_name  + "/" + training_name + "_best_dist_vs_episodes.pdf", 
				bbox_inches = 'tight')
	plt.close()

	plt.title("Success vs Number of episodes")
	plot_3 = sns.lineplot(data = np.array(training_logs['success']))
	plt.ylabel("Success")
	plt.xlabel("Number of episodes")
	plt.savefig("Training_Plots/" + training_name + "/" + training_name + "_success_vs_episodes.pdf", 
				bbox_inches = 'tight')
	plt.close()

	plt.title("Frequency of non-success episode termination")
	data = [sum(training_logs['fall']), sum(training_logs['backward']), sum(training_logs['sideways']), sum(training_logs['time_limit'])]
	plot_4 = plt.bar(x = np.arange(len(data)), height = data, tick_label = ['fall', 'backward', 'sideways', 'time_limit'])
	plt.ylabel("Frequency")
	plt.xlabel("Method of non-success episode termination")
	plt.savefig("Training_Plots/" + training_name + "/" + training_name + "_freq_termination.pdf", 
				bbox_inches = 'tight')
	plt.close()

	plt.title("Training Loss vs Time Steps")
	plot_5 = sns.lineplot(data = np.array(nn_loss['nn_training_loss']))
	plt.ylabel("Training Loss")
	plt.yscale('log')
	plt.xlabel("Number of time steps")
	plt.savefig("Training_Plots/" + training_name + "/" + training_name + "_training_loss.pdf", 
				bbox_inches = 'tight')
	plt.close()

if __name__ == "__main__":
	# conduct the post processing for the case when this script is called manually
	post_process(training_name)