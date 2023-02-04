'''
for loss functions and other useful stuff
'''
import math

from turbograd.engine_ import Value

def detach( grad_outputs ):
    if type(grad_outputs) == list:
        return [ gradout.data  for  gradout in grad_outputs]
    if type(grad_outputs) == Value:
        return grad_outputs.data

def MSE(label, pred):
    return (sum(( ygt - yout)**2 for ygt, yout in zip(label, pred)))/ len(pred)

def logistic_cross_entropy(label, pred):
    return sum(-((1-label) * math.log(1 - pred) + label * math.log(pred)))