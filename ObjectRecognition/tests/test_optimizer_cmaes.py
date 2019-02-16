import cma

from ObjectRecognition.optimizer import CMAEvolutionStrategyWrapper

cmaes_opt = CMAEvolutionStrategyWrapper(8, save_path='results/cmaes_soln/test')
cmaes_opt.optimize(cma.ff.rosen)
