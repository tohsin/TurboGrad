from turbograd.nn_ import Neuron
from turbograd.optim import Optimiser
neuron = Neuron(3, '')
optimiser = Optimiser('SGD', lr = 0.01, model = neuron)
x = [ 1, 2, 3]
add = neuron(x)


mse = (add - 6) ** 2
print(mse)
optimiser.zero_grad()
# neuron.zero_grad()
mse.backward()
v = neuron.parameters()
optimiser.step(0.01)
optimiser.zero_grad()
# neuron.zero_grad()
mse.backward()
optimiser.step(0.01)
