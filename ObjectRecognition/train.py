import numpy as np
import torch

import logging
logging.basicConfig(format='%(levelname)s [%(asctime)s]: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
import time


# placeholder for recognition_train
class Trainer():
    def __init__(self, dataset, model, loss_fn, ret_both=False, verbose=False):
        self.dataset = dataset
        self.model = model
        self.loss_fn = loss_fn
        self.verbose = verbose
        self.ret_both = ret_both

    def evaluate_model(self, params):
        self.model.set_parameters(params)
        # start = time.time()
        focus = self.model.forward_all(self.dataset, batch_size=500, 
                                       ret_extra=self.ret_both)
        # print("forward", time.time() - start)
        loss = self.loss_fn(focus)
        # print("loss", time.time() - start)
        if self.verbose:
            logger.info('loss evaluated= %f\n', loss)
        return loss

    def reset(self):
        self.dataset.reset()


def recognition_train(dataset, model, loss_fn, optimizer, 
                      ret_both=False, verbose=False):
    """
    Train model according to conceptual object loss function
    :param dataset:     game environment interface, see dataset.py
    :param model:       model template with constant number of parameters
    :param loss_fn:     loss function with forward method
    :param optimizer:   optimizer with method optimize(callback)
    """
    trainer = Trainer(dataset, model, loss_fn, 
                      ret_both=ret_both, verbose=verbose)
    return optimizer.optimize(trainer.evaluate_model, clear_fn=trainer.reset)