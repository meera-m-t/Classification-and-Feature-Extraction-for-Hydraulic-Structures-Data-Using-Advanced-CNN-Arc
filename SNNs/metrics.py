import numpy as np
import torch

class Metric:
    def __init__(self):
        pass

    def __call__(self, outputs, target, loss):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def value(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError



class AccumulatedAccuracyMetric(Metric):
    """
    Works with classification model
    """

    def __init__(self):
        self.correct = 0
        self.total = 0

    def __call__(self, outputs, target, loss):
        pred = outputs[0].data.max(1, keepdim=True)[1]
        self.correct += pred.eq(target[0].data.view_as(pred)).cpu().sum()
        self.total += target[0].size(0)
        return self.value()

    def reset(self):
        self.correct = 0
        self.total = 0

    def value(self):
        return 100 * float(self.correct) / self.total

    def name(self):
        return 'Accuracy'



# class AccumulatedAccuracyMetric(Metric):
#     """
#     Works with classification model
#     """

#     def __init__(self):
#         self.correct = 0
#         self.total = 0

#     def __call__(self, outputs, target, loss):
  
#         pred1 = outputs[0].data.max(1, keepdim=True)[1]

#         pred2 = outputs[1].data.max(1, keepdim=True)[1]

#         pred=(pred2 - pred1).pow(2).sum(1).float()  # squared distances
# #         print(pred)  
# #         pred=(torch.sigmoid(pred))

#         self.correct += torch.eq(torch.round(pred).type(target[0].type()), target[0]).view(-1).sum()
        
#         self.total += target[0].size(0)
#         return self.value()

#     def reset(self):
#         self.correct = 0
#         self.total = 0

#     def value(self):
#         return 100-( 100 * float(self.correct) / self.total)

#     def name(self):
#         return 'Accuracy'
class AverageNonzeroTripletsMetric(Metric):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''

    def __init__(self):
        self.values = []

    def __call__(self, outputs, target, loss):
        self.values.append(loss[1])
        return self.value()

    def reset(self):
        self.values = []

    def value(self):
        return np.mean(self.values)

    def name(self):
        return 'Average nonzero triplets'
    


