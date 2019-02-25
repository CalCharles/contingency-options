import dopamine

from dopamine.dqn_agent import DQNAgent

class DQNWrapper(Model):
    def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=None, sess=None):
        super(DQNWrapper, self).__init__(args, num_inputs, num_outputs, name=name, factor=factor, minmax=minmax, sess=None)
        self.sess = tf.Session('', config=tf.ConfigProto(allow_soft_placement=True))
        self.dope_dqn = DQNAgent(self.sess,
               num_outputs,
               observation_shape=(num_inputs, ),
               observation_dtype=tf.uint8,
               stack_size=1,
               network=gym_lib._basic_discrete_domain_network,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=10000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               tf_device='/cpu:*',
               use_staging=True,
               max_tf_checkpoints_to_keep=4,
               optimizer=tf.train.RMSPropOptimizer(
                   learning_rate=0.00025,
                   decay=0.95,
                   momentum=0.0,
                   epsilon=0.00001,
                   centered=True),
               summary_writer=None,
               summary_writing_frequency=500)

    def forward(self, x, reward):
        '''
        TODO: make use of time_estimator, link up Q vals and action probs
        TODO: clean up cuda = True to something that is actually true
        '''
        x = pytorch_model.unwrap(x)
        return self.dope_dqn.step(reward, x)

class RainbowWrapper(Model):
    def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=None, sess=None):
        super(RainbowWrapper, self).__init__(args, num_inputs, num_outputs, name=name, factor=factor, minmax=minmax, sess=None)
        self.sess = tf.Session('', config=tf.ConfigProto(allow_soft_placement=True))
        self.dope_rainbow = RainbowAgent(self.sess,
               num_actions,
               observation_shape=(num_inputs, ),
               observation_dtype=tf.float,
               stack_size=1,
               network=gym_lib._basic_discrete_domain_network,
               num_atoms=51,
               vmax=10.,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               replay_scheme='prioritized',
               tf_device='/cpu:*',
               use_staging=True,
               optimizer=tf.train.AdamOptimizer(
                   learning_rate=0.00025, epsilon=0.0003125),
               summary_writer=None,
               summary_writing_frequency=500):

    def forward(self, x, reward):
        '''
        TODO: make use of time_estimator, link up Q vals and action probs
        TODO: clean up cuda = True to something that is actually true
        '''
        x = pytorch_model.unwrap(x)
        return self.dope_rainbow.step(reward, x)


class ImplicitQuantileAgent(Model):
    def __init__(self, args, num_inputs, num_outputs, name="option", factor=8, minmax=None, sess=None):
        super(ImplicitQuantileAgent, self).__init__(args, num_inputs, num_outputs, name=name, factor=factor, minmax=minmax, sess=None)
        self.sess = tf.Session('', config=tf.ConfigProto(allow_soft_placement=True))
        self.dope_iq = ImplicitQuantileAgent(self.sess,
               num_actions,
               observation_shape=(num_inputs, ),
               observation_dtype=tf.float,
               stack_size=1,
               network=gym_lib._basic_discrete_domain_network,
               num_atoms=51,
               vmax=10.,
               gamma=0.99,
               update_horizon=1,
               min_replay_history=20000,
               update_period=4,
               target_update_period=8000,
               epsilon_fn=dqn_agent.linearly_decaying_epsilon,
               epsilon_train=0.01,
               epsilon_eval=0.001,
               epsilon_decay_period=250000,
               replay_scheme='prioritized',
               tf_device='/cpu:*',
               use_staging=True,
               optimizer=tf.train.AdamOptimizer(
                   learning_rate=0.00025, epsilon=0.0003125),
               summary_writer=None,
               summary_writing_frequency=500,
               kappa=1.0,
               num_tau_samples=32,
               num_tau_prime_samples=32,
               num_quantile_samples=32,
               quantile_embedding_dim=64,
               double_dqn=False):

    def forward(self, x, reward):
        '''
        TODO: make use of time_estimator, link up Q vals and action probs
        TODO: clean up cuda = True to something that is actually true
        '''
        x = pytorch_model.unwrap(x)
        return self.dope_iq.step(reward, x)