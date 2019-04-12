import dopamine
import tensorflow as tf
import cv2
slim = tf.contrib.slim
from dopamine.discrete_domains import gym_lib, atari_lib
from ReinforcementLearning.models import Model, pytorch_model


from dopamine.agents.dqn import dqn_agent
from dopamine.agents.rainbow import rainbow_agent
from tensorflow.python.client import device_lib


def create_dqn_network(minmax):
  def fc_dqn_network(num_actions, network_type, state):
    """The convolutional network used to compute the agent's Q-values.
    Args:
    num_actions: int, number of actions.
    network_type: namedtuple, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.
    Returns:
    net: _network_type object containing the tensors output by the network.
    """
    q_values = gym_lib._basic_discrete_domain_network(
      pytorch_model.unwrap(minmax[0]), pytorch_model.unwrap(minmax[1]), num_actions, state)
    return network_type(q_values)
  return fc_dqn_network


class DQNWrapper(Model):
  def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=None, sess=None):
    super(DQNWrapper, self).__init__(args, num_inputs, num_outputs, name=name, factor=factor, minmax=minmax, sess=None)
    self.sess = tf.Session('', config=tf.ConfigProto(allow_soft_placement=True))
    self.dope_dqn = dqn_agent.DQNAgent(self.sess,
           num_outputs,
           observation_shape=(num_inputs, ),
           observation_dtype=tf.float32,
           stack_size=1,
           network=create_dqn_network(self.minmax),
           gamma=args.gamma,
           update_horizon=3,
           min_replay_history=10000,
           update_period=4,
           target_update_period=8000,
           epsilon_fn=dqn_agent.linearly_decaying_epsilon,
           epsilon_train=args.greedy_epsilon,
           epsilon_eval=0.001,
           epsilon_decay_period=250000,
           tf_device='/cpu:*',
           use_staging=True,
           max_tf_checkpoints_to_keep=4,
           optimizer=tf.train.RMSPropOptimizer(
               learning_rate=args.lr,
               decay=0.95,
               momentum=0.0,
               epsilon=args.eps,
               centered=True),
           summary_writer=None,
           summary_writing_frequency=500)
    init = tf.global_variables_initializer()
    self.sess.run(init)


  def forward(self, x, reward):
    '''
    TODO: make use of time_estimator, link up Q vals and action probs
    TODO: clean up cuda = True to something that is actually true
    '''
    x = pytorch_model.unwrap(x)
    # print(reward, x)
    return self.dope_dqn.step(reward[0], x)

  def begin_episode(self, observation):
    self.dope_dqn.begin_episode(observation)

  def end_episode(self, reward):
    self.dope_dqn.end_episode(reward[0])


def create_rainbow_network(minmax):
  def fc_rainbow_network(num_actions, num_atoms, support, network_type,
                               state):
    """Build the deep network used to compute the agent's Q-value distributions.
    Args:
    num_actions: int, number of actions.
    num_atoms: int, the number of buckets of the value function distribution.
    support: tf.linspace, the support of the Q-value distribution.
    network_type: `namedtuple`, collection of expected values to return.
    state: `tf.Tensor`, contains the agent's current state.
    Returns:
    net: _network_type object containing the tensors output by the network.
    """
    print(minmax)
    net = gym_lib._basic_discrete_domain_network(
      pytorch_model.unwrap(minmax[0]), pytorch_model.unwrap(minmax[1]), num_actions, state,
      num_atoms=num_atoms)
    logits = tf.reshape(net, [-1, num_actions, num_atoms])
    probabilities = tf.contrib.layers.softmax(logits)
    q_values = tf.reduce_sum(support * probabilities, axis=2)
    return network_type(q_values, logits, probabilities)
  return fc_rainbow_network


class RainbowWrapper(Model):
  def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=None, sess=None):
    super(RainbowWrapper, self).__init__(args, num_inputs, num_outputs, name=name, factor=factor, minmax=minmax, sess=None)
    self.sess = tf.Session('', config=tf.ConfigProto(allow_soft_placement=True))
    # local_device_protos = device_lib.list_local_devices()
    # print([x.name for x in local_device_protos])
    # print(atari_lib.NATURE_DQN_OBSERVATION_SHAPE,
    #            atari_lib.NATURE_DQN_DTYPE,
    #            atari_lib.NATURE_DQN_STACK_SIZE,)
    # print(num_inputs)
    self.dope_rainbow = rainbow_agent.RainbowAgent(self.sess,
           num_outputs,
           observation_shape=(num_inputs, ),
           observation_dtype=tf.float32,
           stack_size=1,
           network=create_rainbow_network(self.minmax),
           num_atoms=51,
           vmax=args.value_bounds[1],
           gamma=args.gamma,
           update_horizon=3,
           min_replay_history=args.buffer_steps,
           update_period=4,
           target_update_period=8000,
           epsilon_fn=dqn_agent.linearly_decaying_epsilon,
           epsilon_train=args.greedy_epsilon,
           epsilon_eval=0.001,
           epsilon_decay_period=250000,
           replay_scheme='prioritized',
           tf_device='/gpu:*',
           use_staging=True,
           optimizer=tf.train.AdamOptimizer(
               learning_rate=args.lr, epsilon=0.0003125),
           summary_writer=None,
           summary_writing_frequency=500)
    self.sess.run(tf.global_variables_initializer())

  def forward(self, x, reward):
    x = pytorch_model.unwrap(x)
    return self.dope_rainbow.step(reward[0], x)

  def begin_episode(self, observation):
    self.dope_rainbow.begin_episode(observation)

  def end_episode(self, reward):
    self.dope_rainbow.end_episode(reward[0])


# class ImplicitQuantileAgent(Model):
#     def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=None, sess=None):
#         super(ImplicitQuantileAgent, self).__init__(args, num_inputs, num_outputs, name=name, factor=factor, minmax=minmax, sess=None)
#         self.sess = tf.Session('', config=tf.ConfigProto(allow_soft_placement=True))
#         self.dope_iq = ImplicitQuantileAgent(self.sess,
#                num_actions,
#                observation_shape=(num_inputs, ),
#                observation_dtype=tf.float,
#                stack_size=1,
#                network=gym_lib._basic_discrete_domain_network,
#                num_atoms=51,
#                vmax=10.,
#                gamma=0.99,
#                update_horizon=1,
#                min_replay_history=20000,
#                update_period=4,
#                target_update_period=8000,
#                epsilon_fn=dqn_agent.linearly_decaying_epsilon,
#                epsilon_train=0.01,
#                epsilon_eval=0.001,
#                epsilon_decay_period=250000,
#                replay_scheme='prioritized',
#                tf_device='/cpu:*',
#                use_staging=True,
#                optimizer=tf.train.AdamOptimizer(
#                    learning_rate=args.lr, epsilon=args.eps),
#                summary_writer=None,
#                summary_writing_frequency=500,
#                kappa=1.0,
#                num_tau_samples=32,
#                num_tau_prime_samples=32,
#                num_quantile_samples=32,
#                quantile_embedding_dim=64,
#                double_dqn=False):

#     def forward(self, x, reward):
#         '''
#         TODO: make use of time_estimator, link up Q vals and action probs
#         TODO: clean up cuda = True to something that is actually true
#         '''
#         x = pytorch_model.unwrap(x)
#         return self.dope_iq.step(reward, x)

models = {"rainbow": RainbowWrapper, "DQN": DQNWrapper}