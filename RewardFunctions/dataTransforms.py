import numpy as np
from Environments.state_definition import midpoint_separation

    
# argument options for selecting transforms

def define_proximal(x, closeness_threshold = 2):
    if x < 2:
        return True
    return False

class InputTransformer():
    def format_function(self, models, changepoints, correlate_trajectory, trajectory, window):
        raise NotImplementedError

    def mode_statistics(self, models, changepoints, trajectory=[], values=[], window=-1, total=-1):
        raise NotImplementedError


class SegmentTransform(InputTransformer):
    def __init__(self):
        self.segment = True

class ParameterTransform(SegmentTransform):
    def format_function(self, models, changepoints, correlate_trajectory, trajectory, window=None):
        return models[0].parameters().flatten()

    def mode_statistics(self, models, changepoints=[], trajectory=[], values=[], window=-1):
        model_parameters = [m.A.flatten().tolist() for m in models]
        data = np.array(model_parameters)
        return data

class VelocitySegmentTransform(SegmentTransform):
    def format_function(self, models, changepoints, correlate_trajectory, trajectory, window=None):
        values = trajectory[changepoints[0]:changepoints[1] + 1]
        values = values[1:] - values[:-1]
        stats = np.sum(values, axis = 0)/float(len(values))
        return stats

    def mode_statistics(self, models, changepoints, trajectory=[], values=[], window=-1):
        '''
        values should have at least one more than total.
        gathers changepoint data which is the average of deltas within a segment
        '''
        total = len(trajectory)
        changepoints.append(total)
        values = trajectory[1:] - trajectory[:-1]
        c_d = {changepoints[i]: values[changepoints[i]:changepoints[i+1]] for i in range(len(changepoints)-1)}
        changepoint_statistics = {key: np.sum(c_d[key], axis = 0)/float(len(c_d[key])) for key in c_d.keys()}
        dat = [(key, val) for key, val in changepoint_statistics.items()]
        dat.sort(key=lambda x: x[0])
        data = np.array([val for key,val in dat])
        changepoints.pop(-1)
        return data


class SegmentAccelerationTransform(SegmentTransform):
    def format_function(self, models, changepoints, correlate_trajectory, trajectory, window=None):
        # ignores spurious changepoints
        values1 = trajectory[changepoints[0]:changepoints[1] + 1]
        values1 = values1[1:] - values1[:-1]
        values2 = trajectory[changepoints[1]:changepoints[2] + 1]
        values2 = values2[1:] - values2[:-1]
        stats1 = np.sum(values1, axis = 0)/float(len(values1))
        stats2 = np.sum(values2, axis = 0)/float(len(values2))
        return stats2 - stats1

    def mode_statistics(self, models, changepoints, trajectory=[], values=[], window=-1):
        '''
        takes the delta of changepoint statistics at a changepoint
        '''
        total = len(trajectory)
        changepoints.append(total)
        values = values[1:] - values[:-1]
        c_d = {changepoints[i]: values[changepoints[i]:changepoints[i+1]] for i in range(len(changepoints)-1)}
        changepoint_statistics = {key: np.sum(c_d[key], axis = 0)/float(len(c_d[key])) for key in c_d.keys()}
        dat = [(key, val) for key, val in changepoint_statistics.items()]
        dat.sort(key=lambda x: x[0])
        data = []
        for i in range(1,len(dat[1:-1])):
            lkey, lval = dat[i-1]
            key, val = dat[i]
            data.append((val-lval).tolist())
        data = np.array(data)
        changepoints.pop(-1)
        return data


class SegmentCorrelateAverageTransform(SegmentTransform):
    def format_function(self, models, changepoints, correlate_trajectory, trajectory, window=None):
        correlate = correlate_trajectory[changepoints[1]:changepoints[2]]
        stats = np.sum(correlate, axis = 0)/float(len(correlate))
        return stats

    def mode_statistics(self, models, changepoints, trajectory=[], correlate=[], window=-1):
        total = len(trajectory)
        changepoints.append(total)
        c_d = {changepoints[i]: correlate[changepoints[i]:changepoints[i+1]] for i in range(len(changepoints)-1)}
        changepoint_statistics = {key: np.sum(c_d[key], axis = 0)/len(c_d[key]) for key in c_d.keys()}
        dat = [(key, val) for key, val in changepoint_statistics.items()]
        dat.sort(key=lambda x: x[0])
        data = np.array([val for key,val in dat])
        changepoints.pop(-1)
        return data
    

class WindowTransform(InputTransformer):
    def __init__(self):
        self.segment = False

class WindowPositionTransform(WindowTransform):
    def format_function(self, models, changepoints, correlate_trajectory, trajectory, window):
        v1 = trajectory[max(0,changepoints[1]-window+1):changepoints[1]+1]
        v2 = trajectory[changepoints[1]+1:min(len(trajectory),changepoints[1]+window+1)]
        vshape = trajectory[0].shape
        if len(v1) == 0:
            v1 = np.zeros(vshape)
        else:
            v1 = np.sum(v1, axis=0)
        if len(v2) == 0:
            v2 = np.zeros(vshape)
        else:
            v2 = np.sum(v2, axis=0)
        return np.array(v1.tolist() + v2.tolist()).flatten()

    def mode_statistics(self, models, changepoints, trajectory=[], values=[], window=-1):
        '''
        takes the delta of changepoint statistics at a changepoint
        TODO: clipping may help identify clusters
        '''
        vshape = values[0].shape
        c_d = {changepoints[i]: [values[max(0,changepoints[i]-window+1):changepoints[i]+1], values[changepoints[i]+1:min(total,changepoints[i]+window+1)]] for i in range(len(changepoints)-1)}

        changepoint_statistics = dict()
        for key, val in c_d.items():
            v1 = val[0]
            v2 = val[1]
            if len(v1) == 0:
                v1 = np.zeros(vshape)
            else:
                v1 = np.sum(v1, axis=0)
            if len(v2) == 0:
                v2 = np.zeros(vshape)
            else:
                v2 = np.sum(v2, axis=0)
            changepoint_statistics[key] = [v1, v2]
        dat = [(key, val) for key, val in changepoint_statistics.items()]
        dat.sort(key=lambda x: x[0])
        data = []
        for i in range(1,len(dat[1:-1])):
            data.append(np.array(dat[i][0].tolist() + dat[i][0].tolist()).flatten().tolist())
        data = np.array(data)
        changepoints.pop(-1)
        return data

class WindowCorrelateAverageTransform(WindowTransform):
    def format_function(self, models, changepoints, correlate_trajectory, trajectory, window):
        correlate = correlate_trajectory[max(0,changepoints[1]-window+1):min(len(trajectory),changepoints[1]+window+1)]
        return np.sum(correlate, axis = 0)/float(len(correlate))

    def mode_statistics(self, models, changepoints, trajectory=[], correlate=[], window=-1):
        total = len(trajectory)
        c_d = {changepoints[i]: correlate[max(0,changepoints[i]-window):min(total, changepoints[i]+window+1)] for i in range(len(changepoints))}
        changepoint_statistics = {key: np.sum(c_d[key], axis = 0)/len(c_d[key]) for key in c_d.keys()}
        dat = [(key, val) for key, val in changepoint_statistics.items()]
        dat.sort(key=lambda x: x[0])
        data = np.array([val for key,val in dat])
        return data


class WindowCorrelateProximityPostVelocity(WindowTransform):
    '''
    This takes the velocity statistics of the object only after a proximal occurrance. 
    '''
    def format_function(self, models, changepoints, correlate_trajectory, trajectory, window):
        correlate = correlate_trajectory[max(0,changepoints[1]-window+1):min(len(trajectory),changepoints[1]+window+1)]
        values = trajectory[changepoints[1]:min(changepoints[i] + window+5, changepoints[i+1], len(trajectory))]
        values = values[1:] - values[:-1]
        stats = np.median(values, axis = 0)
        return (correlate[np.argmin(np.abs(c_d[key][:,0]))], stats)

    def mode_statistics(self, models, changepoints, trajectory=[], correlate=[], window=-1):
        total = len(trajectory)
        # models is not models here, but the actual correlate data
        c_d = {changepoints[i]: correlate[max(0,changepoints[i]-window):min(total, changepoints[i]+window+1)] for i in range(len(changepoints))}
        changepoint_statistics = {key: c_d[key][np.argmin(np.abs(c_d[key][:,0]))] for key in c_d.keys()}
        s_i = dict()
        for i in range(len(changepoints)-1):
            s_i[changepoints[i]] = trajectory[changepoints[i] + 1:min(changepoints[i] + window+5, changepoints[i+1])] - trajectory[changepoints[i]:min(changepoints[i] + window+4, changepoints[i+1]-1)]
        s_i[changepoints[-1]] = trajectory[changepoints[-1] + 1:changepoints[-1] + window+5] - trajectory[changepoints[-1]:min(changepoints[-1] + window+4, len(trajectory)-1)]
        # s_i = {changepoints[i]: trajectory[changepoints[i] + 1:min(changepoints[i] + window+5, changepoints[i+1])] - trajectory[changepoints[i]:min(changepoints[i] + window+4, len(trajectory)-1)] for i in range(len(changepoints))} # let's assume this is the right data for now, +5 is hard coded
        # behavior_statistics = {key: np.sum(s_i[key], axis = 0)/max(len(s_i[key]), 1) for key in s_i.keys()}    
        behavior_statistics = {key: np.median(s_i[key], axis = 0) for key in s_i.keys()}    
        dat = [(key, (val, vv)) for ((kv, vv), (key, val)) in zip(behavior_statistics.items(), changepoint_statistics.items())]
        dat.sort(key=lambda x: x[0])
        data = np.array([v1 for key,(v1, v2) in dat if define_proximal(v2)])#, np.array([v2 for key,(v1, v2) in dat])
        return data

class WindowPostVelocity(WindowTransform):
    '''
    This takes the velocity statistics of the object only after a proximal occurrance. 
    '''
    def format_function(self, models, changepoints, correlate_trajectory, trajectory, window):
        values = trajectory[changepoints[1]:min(changepoints[i] + window+5, changepoints[i+1], len(trajectory))]
        values = values[1:] - values[:-1]
        stats = np.median(values, axis = 0)
        return stats

    def mode_statistics(self, models, changepoints, trajectory=[], correlate=[], window=-1):
        total = len(trajectory)
        # models is not models here, but the actual correlate data
        s_i = dict()
        for i in range(len(changepoints)-1):
            s_i[changepoints[i]] = trajectory[changepoints[i] + 1:min(changepoints[i] + window+5, changepoints[i+1])] - trajectory[changepoints[i]:min(changepoints[i] + window+4, changepoints[i+1]-1)]
        s_i[changepoints[-1]] = trajectory[changepoints[-1] + 1:changepoints[-1] + window+5] - trajectory[changepoints[-1]:min(changepoints[-1] + window+4, len(trajectory)-1)]
        # s_i = {changepoints[i]: trajectory[changepoints[i] + 1:min(changepoints[i] + window+5, changepoints[i+1])] - trajectory[changepoints[i]:min(changepoints[i] + window+4, len(trajectory)-1)] for i in range(len(changepoints))} # let's assume this is the right data for now, +5 is hard coded
        # behavior_statistics = {key: np.sum(s_i[key], axis = 0)/max(len(s_i[key]), 1) for key in s_i.keys()}    
        behavior_statistics = {key: np.median(s_i[key], axis = 0) for key in s_i.keys()}    
        dat = [(key, val) for key, val in behavior_statistics.items()]
        dat.sort(key=lambda x: x[0])
        data = np.array([v1 for key, v1 in dat])#, np.array([v2 for key,(v1, v2) in dat])
        return data


class WindowCorrelateProximity(WindowTransform):
    def format_function(self, models, changepoints, correlate_trajectory, trajectory, window):
        correlate = [midpoint_separation((t,c)) for t,c in zip(trajectory, correlate)]
        correlate = correlate_trajectory[max(0,changepoints[1]-window+1):min(len(trajectory),changepoints[1]+window+1)]
        return correlate[np.argmin(np.abs(c_d[key][:,0]))]

    def mode_statistics(self, models, changepoints, trajectory=[], correlate=[], window=-1):
        total = len(trajectory)
        correlate = [midpoint_separation((t,c)) for t,c in zip(trajectory, correlate)]
        c_d = {changepoints[i]: correlate[max(0,changepoints[i]-window):min(total, changepoints[i]+window+1)] for i in range(len(changepoints))}
        changepoint_statistics = {key: [c_d[key][np.argmin(np.sum(np.abs(c_d[key]), axis = 1))]][0] for key in c_d.keys()}
        # print(changepoint_statistics)
        dat = [(key, val) for key, val in changepoint_statistics.items()]
        dat.sort(key=lambda x: x[0])
        data = np.array([val for key,val in dat])
        return data

arg_transform = {"param": ParameterTransform, "SVel": VelocitySegmentTransform, "SAcc": SegmentAccelerationTransform, 
        "SCorAvg": SegmentCorrelateAverageTransform, "WPos": WindowPositionTransform, "WCorAvg": WindowCorrelateAverageTransform,
        "WVel": WindowPostVelocity, "WProx": WindowCorrelateProximity}








