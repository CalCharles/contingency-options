# contingency-options

Description of folders and files:
  Behavior Policies: contains policies that take in action probabilities or Q values, and returns actions to be taken, and files that might be useful for this
    behavior_policies.: contains some basic behavior policies
  ChangepointDetection: contains changepoint detector and related files
    ChangepointDetectorBase contains the base class for changepoint functions
    CHAMP contains an implementation of CHAMP algorithm
    DynamicsModels contains linear dyanmics models, used by CHAMP
  Environments: contains environments, which take in actions and retain evolving state
    environment_specification: contains different environments, and Proxy-environment constructed with options
    state_definition: contains information about state and how to extract state from environments
    multioption: contains a policy for switching between options as actions
  OptionChain: contains necessary files to manage graph of object relationships
    option_chain: contains the option chain code, option chains are stored in the directory and name structre
  ReinforcementLearning: contains useful files for reinforcement learning
    learning_algorithms: contains optimization operations for different reinforcement learning methods
    models: contains mostly pytorch models for going from state to critic, actor or Q value
    rollouts: contains a class that stores useful information at each step for rolling out a reinforcement learning function
    train_rl: contains code to train a reinforcement learning agent
  RewardFunctions: allows the generation of reward functions from data and objects
    dataTransforms: transforms a trajectory of data (series of n vectors, where n is not fixed) to a fixed length fector
    changepointClusterModel: contains classes for clustering, or performing multiple clusterings on a dataset of transformed windows/segments
    changepointDeterminers: reduces a series of classifications/clusters to option relevant clusters
    changepointCorrelation: a class that can transform a trajectory to a series of cluster classifications
    changepointReward: classes that return reward for a particular changepoint
  SelfBreakout: implementation of the atari game breakout with simplified observations and dynamics
  add_edge: should add an edge to the graph
  rl template: performs RL on a chain MDP
  arguments: contains shared arguments for getting rewards and adding edges
  file_management: contains some functions for saving and getting values
    
