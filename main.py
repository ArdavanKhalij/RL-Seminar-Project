#######################################################################################################################
# Libraries
#######################################################################################################################
import gymnasium as gym
import Zeroth_order as zo
import Simple_population as sp
import Policy_Neural_Network as pnn
import save as s
import plotting as pl
#######################################################################################################################



#######################################################################################################################
# 2. Consider the LunarLanderContinuous task in the OpenAI Gym. You can use any version of the Gym or Gymnasium for
# this (all are available in Pip). It has a simple continuous state (8 floats), and a simple continuous action (2
# floats). The policy is now a neural network with 8 inputs, 128 hidden neurons, and 2 outputs. It “.
# parameters()” attribute allows to get its parameters, as a list of PyTorch tensors (so, this is a list of tensors,
# not one single big tensor).
#######################################################################################################################
# I had to install 'Box2d' in order for LunarLander-v2 to work
env = gym.make("LunarLanderContinuous-v2")

# Number of states
state_size = 8
action_size = 2

# Size of hidden layer in NN
hidden_size = 128

# Number of perturbed samples
N = 10

# Number of episodes for finding best noise
episode_count_for_simple_population = 2

learning_rate = 0.001

perturbation_scale = 1

num_episodes = 1000

# Policy
policy = pnn.Policy(state_size, hidden_size, action_size)

scores1 = []
scores2 = []

scores1 = sp.simple_population(N, env, policy, episode_count_for_simple_population, perturbation_scale, num_episodes)
scores2 = zo.Zeroth_order(policy, env, learning_rate, perturbation_scale, episode_count_for_simple_population, num_episodes)

s.save_list_to_file(scores1, "Data Path Text File of Simple Population")
s.save_list_to_file(scores2, "Data Path Text File of Zeroth Order")

# scores1 = s.read_list_from_file("learning_curve_simple_population_2_episodes.txt")
# scores2 = s.read_list_from_file("learning_curve_zeroth_order_2_episodes.txt")

pl.plot_lists(scores1,
              scores2,
              "Comparing 2 Approaches",
              "Number of Episodes",
              "Mean Reward Sum",
              ['Simple Population', 'Zeroth Order'],
              1)
#######################################################################################################################
