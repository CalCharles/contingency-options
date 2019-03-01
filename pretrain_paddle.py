from SelfBreakout.paddle import Paddle
from SelfBreakout.noblock import PaddleNoBlocks
from arguments import get_args
from Models.pretrain_models import random_actions, pretrain_actions




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
    actions, states, num_actions, state_class  = random_actions(args, true_environment)
    pretrain_actions(args, true_environment, actions, num_actions, state_class, states)