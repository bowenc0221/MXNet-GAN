import mxnet as mx
import numpy as np

# class AccMetric(mx.metric.EvalMetric):
#     def __init__(self):
#         super(AccMetric, self).__init__('Accuracy')
#
#     def update(self, labels, preds):
#         pred = preds[0].asnumpy().ravel()
#         label = labels[0].asnumpy().ravel()
#
#         self.sum_metric += np.sum((pred > 0.5) == label)
#         self.num_inst += len(pred)
#
# class CrossEntropyMetric(mx.metric.EvalMetric):
#     def __init__(self):
#         super(CrossEntropyMetric, self).__init__('CrossEntropy')
#
#     def update(self, labels, preds):
#         pred = preds[0].asnumpy().ravel()
#         label = labels[0].asnumpy().ravel()
#
#         ce = -(label * np.log(pred + 1e-12) + (1. - label) * np.log(1. - pred + 1e-12))
#
#         self.sum_metric += np.sum(ce)
#         self.num_inst += len(pred)

def facc(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return ((pred > 0.5) == label).mean()

def fentropy(label, pred):
    pred = pred.ravel()
    label = label.ravel()
    return -(label * np.log(pred + 1e-12) + (1. - label) * np.log(1. - pred + 1e-12)).mean()
