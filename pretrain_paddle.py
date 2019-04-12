from SelfBreakout.paddle import Paddle
from SelfBreakout.noblock import PaddleNoBlocks
from arguments import get_args
from Models.pretrain_models import random_actions, action_criteria, pretrain, range_Qvals, Q_criteria




if __name__ == "__main__":
    '''
    record rollouts
    changepoint dir
    learning rate
    eps
    betas 
    weight decay
    save interval
    save models
    train edge
    state names
    state forms
    '''
    args = get_args()
    true_environment = Paddle()
    if args.optimizer_form in ["PPO", "A2C", "PG"]:
        actions, states, resps, num_actions, state_class  = random_actions(args, true_environment)
        criteria = action_criteria
    elif args.optimizer_form in ["DQN", "SARSA", "Dist"]: # dist might have mode collapse to protect against
        actions, states, resps, num_actions, state_class  = range_Qvals(args, true_environment, [.4, .6]) # TODO: hardcoded range of Q values
        criteria = Q_criteria

    pretrain(args, true_environment, actions, num_actions, state_class, states, resps, criteria)