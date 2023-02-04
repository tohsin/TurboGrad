import random 
from turbograd.engine_ import Value
from turbograd.nn_ import param
class Module:
    def step(self, lr):
        # for p in self.parameters():
        #     p.data += -lr * p.grad
        
        # for p in self.parameters():
        #     if p.type == 'weight':
        #         p.value.data += -lr * p.value.grad
        #     elif p.type == 'bias':
        #         p.value.data += -lr * p.value.grad
        return NotImplementedError
    def zero_grad(self):
        return NotImplementedError

    def parameters(self) -> list[param]:
        return []
    
class param:
    def __init__(self) -> None:
        self.name = None
        self.type = None
        self.Vdw = None
        self.Sdw = None
        self.Vdb = None
        self.Sdb = None

class Weight(param):
    def __init__(self) -> None:
        self.type = 'weight'
        self.value = Value(random.uniform(-1, 1)) 
        self.Vdw = 0
        self.Sdw = 0

class Bias(param):
    def __init__(self) -> None:
        self.type = 'bias'
        self.value = Value(random.uniform(-1, 1)) 
        self.Vdb = 0
        self.Sdb = 0

class Neuron(Module):
    def __init__(self, nin, activation) -> None:
        self.activation = activation
        self.weights = [Weight() for _ in range(nin)]

        self.bias : Bias = Bias()
    def __call__(self, x):
        act = sum( (wi.value * xi for wi, xi in zip(self.weights, x)), self.bias.value) 
        if self.activation == 'tanh':
            out = act.tanh()
        elif self.activation == 'relu':
            out = act.relu()
        elif self.activation == 'sigmoid':
            out = act.sigmoid()
        elif self.activation == 'softmax':
            return NotImplementedError
        else:
            out = act
        return out

    def parameters(self):
        return self.weights + [self.bias]
        # return self.w + [self.b]

class Layer(Module):
    def __init__(self, nin, nout, activation) -> None:
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs


    def parameters(self):
        params = [p for neuron in self.neurons for p in neuron.parameters()]
        return params

# this might have to be re written any time or can be inheritted
class MLP(Module):
    def __init__(self, nin, nouts) -> None:
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    



