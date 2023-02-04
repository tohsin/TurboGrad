# import random 
# from turbograd.engine_ import Value

# class Module:
#     def step(self, lr):
#         for p in self.parameters():
#             p.data += -lr * p.grad
#     def zero_grad(self):
#         for p in self.parameters():
#             p.grad = 0

#     def parameters(self):
#         return []


# class Neuron(Module):
#     def __init__(self, nin, activation) -> None:
#         self.activation = activation
#         self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
#         self.b = Value(random.uniform(-1, 1))

#     def __call__(self, x):
#         act = sum( (wi*xi for wi, xi in zip(self.w, x)), self.b) 
#         if self.activation == 'tanh':
#             out = act.tanh()
#         elif self.activation == 'relu':
#             out = act.relu()
#         elif self.activation == 'sigmoid':
#             out = act.sigmoid()
#         elif self.activation == 'softmax':
#             return NotImplementedError
#         else:
#             out = act
#         return out

#     def parameters(self):
#         return self.w + [self.b]

# class Layer(Module):
#     def __init__(self, nin, nout, activation) -> None:
#         self.neurons = [Neuron(nin, activation) for _ in range(nout)]

#     def __call__(self, x):
#         outs = [n(x) for n in self.neurons]
#         return outs[0] if len(outs) ==1 else outs


#     def parameters(self):
#         params = [p for neuron in self.neurons for p in neuron.parameters()]

#         return params

# # this might have to be re written any time or can be inheritted
# class MLP(Module):
#     def __init__(self, nin, nouts) -> None:
#         sz = [nin] + nouts
#         self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]

#     def __call__(self, x):
#         for layer in self.layers:
#             x = layer(x)
#         return x
#     def parameters(self):
#         return [p for layer in self.layers for p in layer.parameters()]