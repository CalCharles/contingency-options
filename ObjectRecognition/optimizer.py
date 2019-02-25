from __future__ import division, absolute_import, print_function
import os
import torch
import numpy as np
import cma
import pickle
from multiprocessing import Pool

import logging
logging.basicConfig(format='%(levelname)s [%(asctime)s]: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

import ObjectRecognition.util as util


class OptimizerInterface:

    # optimize objective function
    def optimize(self, objective_fct):
        raise NotImplementedError


    # reset optimizer state
    def reset(self):
        raise NotImplementedError


"""
wrapper for pycma
cmaes_params details:
    http://cma.gforge.inria.fr/apidocs-pycma/cma.evolution_strategy.CMAEvolutionStrategy.html
"""
class CMAEvolutionStrategyWrapper(OptimizerInterface):

    def __init__(self, dim, *args, **kwargs):
        self.dim = dim
        self.id =  kwargs.get('id', str(np.random.randint(42000, 42999)))
        self.cmaes_params = kwargs.get('cmaes_params', {})
        self.save_path = util.get_dir(kwargs.get('save_path', 
                                      os.path.join('cmaes_soln', self.id)))
        self.max_niter = kwargs.get('max_niter', 100)
        self.nproc = kwargs.get('nproc', 1)
        self.cheating = kwargs.get('cheating', None)
        self.file_id = 0

        # initialize CMA-ES core
        self.reset()

        logger.info('created wrapper with id= %s', self.id)


    # optimize objective function
    def optimize(self, objective_fct, clear_fn=None):
        self.iter_id = 0
        while not self._stop():
            solutions = self.cmaes.ask()
            costs = self._evaluate(objective_fct, solutions)
            self.cmaes.tell(solutions, costs)
            self._report(solutions, costs)
            if clear_fn: clear_fn()
            self.iter_id += 1

        dump_name = '%s_result_cmaes.pkl'%(self.id)
        dump_path = os.path.join(self.save_path, dump_name)
        with open(dump_path, 'wb') as fout:  # TODO: refactors
            pickle.dump(self.cmaes, fout)
            logger.info('saved cmaes instance to %s', dump_path)
        soln = self.cmaes.result.xbest
        soln_path = self._save(soln)
        logger.info('saved best solution to %s', soln_path)
        self.reset()
        return soln


    # reset optimizer state
    def reset(self):
        # TODO: isolate this for concurrent usage?
        # TODO: more config to CMA-ES
        # initialize pycma class
        xinit = np.random.rand(self.dim)-0.5  # [-0.5, 0.5]^n
        if self.cheating:  # for testing filter generality
            xinit = util.cheat_init_center((10, 10), 3, self.cheating) 
            self.cmaes_params['popsize'] = 2
        self.cmaes = cma.CMAEvolutionStrategy(xinit, 1.0, self.cmaes_params)


    # evaluate a list of solutions
    def _evaluate(self, objective_fct, solutions):
        costs = np.zeros((len(solutions),))
        if self.nproc > 1:  # parallel evaluation
            # TODO: fix
            with Pool(processes=self.nproc) as pool:
                costs = list(pool.map(objective_fct, solutions))
        else:  # serial evaluation
            for i, solution in enumerate(solutions):
                costs[i] = objective_fct(solution)
        return costs


    # process visualization
    def _report(self, solutions, costs):
        self.cmaes.disp()

        # save best solution to file
        best_idx = np.argmin(costs)
        file_path = self._save(solutions[best_idx])
        logger.info('saved solution to %s, cost= %f', file_path, costs[best_idx])


    def _save(self, solution):
        file_name = '%s_%d.npy'%(self.id, self.file_id)
        file_path = os.path.join(self.save_path, file_name)
        with open(file_path, 'wb') as fout:
            np.save(fout, solution)
        self.file_id += 1
        return file_path


    # check stopping criteria
    def _stop(self):
        is_max_iter = self.iter_id >= self.max_niter
        return self.cmaes.stop() or is_max_iter