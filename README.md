# contingency-options
Dependencies:
  * CMA: http://cma.gforge.inria.fr/apidocs-pycma/cma.html
  * pytorch: https://pytorch.org/
  * openAI gym
  * tensorflow
Description of folders and files:
  * Behavior Policies: contains policies that take in action probabilities or Q values, and returns actions to be taken, and files that might be useful for this
    * behavior_policies.: contains some basic behavior policies
  * ChangepointDetection: contains changepoint detector and related files
    * ChangepointDetectorBase contains the base class for changepoint functions
    * CHAMP contains an implementation of CHAMP algorithm
    * DynamicsModels contains linear dyanmics models, used by CHAMP
  * Environments: contains environments, which take in actions and retain evolving state
    * environment_specification: contains different environments, and Proxy-environment constructed with options
    * state_definition: contains information about state and how to extract state from environments
    * multioption: contains a policy for switching between options as actions
  * OptionChain: contains necessary files to manage graph of object relationships
    * option_chain: contains the option chain code, option chains are stored in the directory and name structre
  * ReinforcementLearning: contains useful files for reinforcement learning
    * learning_algorithms: contains optimization operations for different reinforcement learning methods
    * models: contains mostly pytorch models for going from state to critic, actor or Q value
    * rollouts: contains a class that stores useful information at each step for rolling out a reinforcement learning function
    * train_rl: contains code to train a reinforcement learning agent
  * RewardFunctions: allows the generation of reward functions from data and objects
    * dataTransforms: transforms a trajectory of data (series of n vectors, where n is not fixed) to a fixed length fector
    * changepointClusterModel: contains classes for clustering, or performing multiple clusterings on a dataset of transformed windows/segments
    * changepointDeterminers: reduces a series of classifications/clusters to option relevant clusters
    * changepointCorrelation: a class that can transform a trajectory to a series of cluster classifications
    * changepointReward: classes that return reward for a particular changepoint
  * SelfBreakout: implementation of the atari game breakout with simplified observations and dynamics
  * add_edge: should add an edge to the graph
  * rl template: performs RL on a chain MDP
  * arguments: contains shared arguments for getting rewards and adding edges
  * file_management: contains some functions for saving and getting values
    
List of commands to perform a full integration operation:
```
  python ObjectRecognition/main_train.py     results/cmaes_soln/focus_self self-test  --n_state 1000  --popsize 30    ObjectRecognition/net_params/two_layer.json     --saliency 0.0 1.0 0.0 0.0  --hinge_dist 0.3    --action_micp 10.0 1.0       --verbose --niter 20 --save-name paddle
  python ObjectRecognition/main_train_smooth.py   results/cmaes_soln/focus_self   self-test  --n_state 1000    none    --save-name paddle  --premise_path results/cmaes_soln/focus_self/paddle_20.npy  --premise_net ObjectRecognition/net_params/two_layer.json --cuda
  python ObjectRecognition/write_focus_dumps.py data/fullrandom results/cmaes_soln/focus_self
  python ChangepointDetection/CHAMP.py --train-edge "Action->Paddle" --record-rollouts data/fullrandom/ --champ-parameters "Paddle" --focus-dumps-name focus_dumps.txt
  python get_reward.py --record-rollouts data/fullrandom/ --changepoint-dir data/fullintegrationgraph/ --train-edge "Action->Paddle" --transforms SVel SCorAvg --determiner overlap --reward-form markov --segment --train --num-stack 2 --focus-dumps-name focus_dumps.txt --gpu 1
  python add_edge.py --model-form basic --optimizer-form DQN --record-rollouts "data/fullrandom/" --train-edge "Action->Paddle" --changepoint-dir data/fullintegrationgraph --num-stack 2 --factor 6 --train --num-iters 1000 --save-dir data/fullpaddle --state-forms bounds --state-names Paddle --num-steps 1 --reward-check 5 --num-update-model 1 --greedy-epsilon .1 --lr 1e-2 --init-form smalluni --behavior-policy egq --grad-epoch 5 --entropy-coef .01 --value-loss-coef 0.5 --gamma .9 --gpu 1 --save-models --save-dir data/fullintegrationpaddle --save-graph data/fullintnetpaddle > fullintegration/paddle.txt
  python ObjectRecognition/main_train.py     results/cmaes_soln/focus_self self-test2  --n_state 1000 --popsize 50    ObjectRecognition/net_params/two_layer.json     --saliency 0.0 0.0 0.0 1.0  --hinge_dist 0.3  --premise_micp 0.0 1.0 1.0 20.0 0.05  --attn_premise_micp 0.0 0.0 0.2 0.1 0.7  --premise_path results/cmaes_soln/focus_self/paddle.pth  --premise_net ObjectRecognition/net_params/attn_softmax.json       --verbose --save-name ball1
  python ObjectRecognition/main_report.py results/cmaes_soln/focus_self/ self-test ObjectRecognition/net_params/two_layer.json --modelID ball1_25 --n_state 1000 --plot-intensity --plot-focus --plot-cp
  python ObjectRecognition/main_train_smooth.py   results/cmaes_soln/focus_self   self-test  --n_state 1000    none    --save-name paddle  --premise_path results/cmaes_soln/focus_self/paddle.pth  --premise_net ObjectRecognition/net_params/attn_softmax.json --cuda --premise_path_2 results/cmaes_soln/focus_self/balltest_40.npy  --premise_net_2 ObjectRecognition/net_params/two_layer.json
  python ObjectRecognition/write_focus_dumps.py data/fullintegrationpaddle results/cmaes_soln/focus_self --ball --cuda
  python ChangepointDetection/CHAMP.py --train-edge "Paddle->Ball" --record-rollouts data/fullintegrationpaddle/ --champ-parameters "Ball" --focus-dumps-name focus_dumps.txt > fullintegration/ballCHAMP.txt
  python get_reward.py --record-rollouts data/fullintegrationpaddle/ --changepoint-dir data/fullintegrationgraph/ --train-edge "Paddle->Ball" --transforms WProx --determiner prox --reward-form changepoint --num-stack 1 --focus-dumps-name focus_dumps.txt --dp-gmm default --period 7
  python add_edge.py --model-form population --optimizer-form CMAES --record-rollouts "data/fullintegrationpaddle/" --train-edge "Paddle->Ball" --num-stack 1 --train --num-iters 30 --state-forms prox vel --state-names Paddle Ball --changepoint-dir ./data/fullintegrationgraph/ --lr 5e-3 --behavior-policy esp --reward-form bounce --gamma .9 --init-form xuni --factor 8 --num-layers 1 --base-form basic --select-ratio .2 --num-population 10 --sample-duration 100 --sample-schedule 12 --warm-up 0 --log-interval 1 --scale 2 --reward-check 10 --gpu 1 --greedy-epsilon .03  --save-models --save-interval 1 --save-graph data/ib2--save-dir data/integrationbounce2 > fullintegration/ball2.txt
  python test_edge.py --model-form population --optimizer-form CMAES --record-rollouts "data/fullintegrationpaddle/" --train-edge "Paddle->Ball" --num-stack 1 --state-forms prox vel bounds bounds --state-names Paddle Ball Paddle Ball --changepoint-dir ./data/fullintegrationgraph/ --behavior-policy esp --gamma .9 --init-form xnorm --num-layers 1 --num-population 10 --sample-duration 100 --warm-up 0 --log-interval 1 --scale 1 --gpu 1 --load-weights --num-iters 30000 --greedy-epsilon .05
```

Training Gripper Domain without visual testing:
```
python Pushing/screen.py data/randompusher --pushgripper
python ChangepointDetection/CHAMP.py --train-edge "Action->Gripper" --record-rollouts data/randompusher/ --champ-parameters "Paddle" --focus-dumps-name object_dumps.txt
python get_reward.py --record-rollouts data/randompusher/ --changepoint-dir data/pushergraph/ --train-edge "Action->Gripper" --transforms SVel SCorAvg --determiner overlap --dp-gmm far --reward-form markov --segment --train --num-stack 2 --focus-dumps-name object_dumps.txt --gpu 1
python add_edge.py --model-form basic --optimizer-form DQN --record-rollouts "data/randompusher/" --train-edge "Action->Gripper" --changepoint-dir data/pushergraph --num-stack 2 --factor 6 --train --num-iters 500 --state-forms bounds --state-names Gripper --num-steps 1 --reward-check 10 --num-update-model 1 --greedy-epsilon .1 --lr 1e-2 --init-form smalluni --behavior-policy egq --grad-epoch 5 --entropy-coef .01 --value-loss-coef 0.5 --gamma .1 --gpu 1 --true-environment --env SelfPusher --frameskip 3 --save-models --save-dir data/gripperdir --save-graph data/grippergraph > pusher/action_gripper.txt
python ChangepointDetection/CHAMP.py --train-edge "Gripper->Block" --record-rollouts data/gripperdir/ --champ-parameters "PaddleLong" --focus-dumps-name object_dumps.txt --dp-gmm Block --period 7
python get_reward.py --record-rollouts data/pushgripper/ --changepoint-dir data/pushergraph/ --train-edge "Gripper->Block" --transforms WProx --determiner prox --reward-form changepoint --num-stack 1 --focus-dumps-name object_dumps.txt --dp-gmm default --period 7
python add_edge.py --model-form vector --optimizer-form PPO --record-rollouts "data/gripperdir/" --train-edge "Gripper->Block" --num-stack 1 --train --num-iters 1000 --state-forms prox bounds bounds --state-names Gripper Gripper Block --env SelfPusher --true-environment --base-node Action --changepoint-dir ./data/pushergraph/ --lr 1e-5 --behavior-policy esp --gamma .99 --init-form xnorm --num-layers 1 --reward-check 128 --changepoint-queue-len 128 --greedy-epsilon .001 --log-interval 10 --num-steps 1 --frameskip 3 --factor 16 --key-dim 2048 --num-grad-states 32 --return-form value --grad-epoch 8 --acti relu --save-dir ../datasets/caleb_data/blockvec --save-graph data/blockvec --save-models
python ChangepointDetection/CHAMP.py --train-edge "Gripper->Block" --record-rollouts ../datasets/caleb_data/blockvec/ --champ-parameters "further" --focus-dumps-name object_dumps.txt --num-iters 10000
python get_reward.py --record-rollouts data/randompusher/ --changepoint-dir data/pushergraph/ --train-edge "Action->Gripper" --transforms SVel SCorAvg --determiner overlap --dp-gmm far --reward-form markov --segment --train --num-stack 2 --focus-dumps-name object_dumps.txt --gpu 1
```