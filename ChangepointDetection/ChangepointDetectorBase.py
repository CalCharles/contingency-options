#Changepoint Detector Base:
import numpy as np
from file_management import get_edge, load_from_pickle, save_to_pickle
# from changepointCorrelation import correlate_data
from SelfBreakout.breakout_screen import read_obj_dumps, get_individual_data, hot_actions


class ChangepointDetector():
    def __init__(self, train_edge):
        self.head,self.tail = get_edge(train_edge)

    def generate_changepoints(self, data, save_dict=False):
        '''
        generates changepoints on data, formatted [num elements, dimension of elements] 
        returns a tuple of: 
            models (a model over the changepoints, TODO: not sure what that should be standardized yet, but at least should contain data in segment)
            changepoints: a vector of values at which changepoints occur. TODO: soft changepoints??
        '''
        pass
        
    def load_obj_dumps(self, args, dumps_name='object_dumps.txt'):
        '''
        Returns the data used for changepoint analysis
        TODO: only uses location information, only uses head data, has fixed range and segment, stores data (not necessary?)
        '''
        obj_dumps = read_obj_dumps(args.record_rollouts, i=-1, rng=100000, dumps_name=dumps_name)
        self.tail_data = [np.array(get_individual_data(tnode, obj_dumps, pos_val_hash=1)) for tnode in self.tail]
        self.head_data = np.array(get_individual_data(self.head, obj_dumps, pos_val_hash=1))
        return self.head_data