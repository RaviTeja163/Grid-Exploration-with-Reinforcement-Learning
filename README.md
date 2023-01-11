# Grid Exploration with Reinforcement Learning

## Introduction

Reinforcement learning is a type of machine learning technique that enables an agent to learn in
an interactive environment by trial and error using feedback from its own actions and
experiences.

The best way to introduce the concepts of reinforcement learning is through games, in this case
we introduce an exploration game which basically introduces an agent to an unknown world.
This agent has to discover the best path to a reward position without knowing the location and
has to avoid a pit of doom and obstacles. All of this is performed in a rectangle bounded
environment.

In order to explore the unknown world we introduce a learning procedure that enables the agent
to understand which are the best steps to take. Our approach introduces a Q learning algorithm
that considers values for every decision.

## Q-learning

We are using the Q-learning algorithm because it helps us maximize the expected reward by
selecting the best of all possible actions, considering initially an exploration of the environment
and then an exploitation of the explored environment.

![image](https://user-images.githubusercontent.com/47236746/211795742-627c15b5-8a19-47e5-b5ec-18de39fffcde.png)

These values are updated using Bellman’s equation that takes two inputs: the state and the action
of the agent. This equation is used to fill a Q-table that helps the agent update its exploration
policy. In this case, S corresponds to state and A for action. R stands for the reward, t denotes the
time step and t+1 denotes the following time step.

As we said earlier Q-learning selects an action based on its reward value. For our approach we
initiate with an epsilon-greedy policy that takes advantage of prior knowledge and exploration to
look for new options.

The aim is to have a balance between exploration and exploitation. Exploration allows us to
have some room for trying new things, sometimes risking our actual knowledge of the
environment.

## Experimentation

We decided to create different codes for each of the parts, namely part1.py, part2.py, part3.py. In
the first part, we used a deterministic model with the greedy epsilon policy approach. For the
second part, we added the probabilities given in the problem statement to form a transition model
and convert our model into a non-deterministic one. We still used the greedy epsilon policy
approach in this.

For the third part, we introduce a new exploration policy that balances the exploration and
exploitation approaches in a better way. This method is called decaying epsilon, which allows
the agent to have a good exploration of the environment at the beginning of the process with a
high probability of exploration and reaches a low probability of exploration once the agent has
become familiar with the environment. The epsilon does not decay with a particular rate, but it
depends on the number of trials completed by the agent. As the trails keep increasing, epsilon
decreases and the exploration decreases.
<p align="center">
<b>e = 1/(trails+1)</b>
</p>
Considering that our approach will be tested with big environments and a short amount of time,
we consider that the best exploration policy for the agent is a decayin epsilon. It gives a good
symmetry in order to find the best reward in the world. Also, this approach allows instant
feedback on the action taken previously, which in this case is an advantage given the time
circumstances of this experiment.

We ran the second part with multiple epsilon values (0.01, 0.1 and 0.3). The algorithm is run 10
times for each of the epsilon values to get an average of the mean rewards obtained. Also for one
of the runs, we plotted a graph of time elapsed vs the mean reward obtained at that point in time.
Then we also run the third part and plot a graph for it as well.

## Wormholes

We implemented the wormholes into the world taking into account the statements given in the
assignment in which if the agent encountered a wormhole it would teletransport into another
position. This makes the learning process harder for the agent given that the exploration becomes
more chaotic and the teletransportation gives a sense of restart because initially the new position
would be completely unknown.

We used the deterministic model and greedy epsilon policy to develop the code for wormholes in
a file extra2.py.
