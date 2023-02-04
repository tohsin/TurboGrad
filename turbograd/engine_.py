# from engine import Value
import math
import random



class Value:
    def __init__(self, data, _children = (), _op = '', label = '') -> None:
        self.data = data
        self.grad = 0.0
        self._backward = lambda : None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self) -> str: 
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
   
   
    def __rmul__(self, other): # other * self
        return self * other

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')
        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward
        return out
        
    def sigmoid(self):
        pass

    def softmax(self):
        x = self.data
        t = 1 / (1 + math.exp(-1 * x))
        out = Value(t, (self,),'softmax')
        def _backward():
            self.grad += t * (1 - t) * out.grad
        out._backward = _backward
        return out
        
    def log(self):
        x = self.data
        out = Value(math.log(x), (self, ),"log")
        def _backward():
            self.grad +=  (1 / x) * out.grad
        out.backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x ) -1) /(math.exp(2*x) + 1)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += ( 1 - (t**2))  * out.grad # grad of tanh is 1 - tanh ^2 x , times global grad due to chain rule

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), "exp")
        def _backward():
            self.grad = out.data * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other ** -1)

    def backward(self):
        # implements a toplogical sort
        toplogical_graph = []
        visited = set ()
        def build_toplogical_graph(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_toplogical_graph(child)
                #finished processing all child nodes
                toplogical_graph.append(v)
        build_toplogical_graph(self)

        self.grad = 1.0
        for node in reversed(toplogical_graph):
            node._backward()

 
    def __neg__(self):
        return self * -1 

    def __sub__(self, other):
        return self + (-other)
    
    def __radd__(self, other): # other + self
        return self + other

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1
