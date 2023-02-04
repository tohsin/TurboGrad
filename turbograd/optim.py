from turbograd.nn_ import Module
import math

class Optimiser(Module):
    def __init__(self, optimiser, lr, model) -> None:
        super().__init__()
        self.model = model
        self.lr = lr
        self.optimiser = optimiser
        self.t = 0
        self.beta1 = 0.9 # momentum
        self.beta2 = 0.999 # rmsprop
      
    def step(self):
        if self.optimiser == 'SGD':
            self.gd()
        elif self.optimiser == 'ADAM':
            self.adamop()

        self.t +=1 

    def set_lr(self, lr):
        self.lr = lr

    def gd(self):
        for p in self.parameters():
            if p.type == 'weight':
                p.value.data += -self.lr * p.value.grad
            elif p.type == 'bias':
                p.value.data += - self.lr * p.value.grad

    def zero_grad(self):
        for p in self.parameters():
            p.value.grad = 0

    def adamoptimiser(self):
      
        for p in self.parameters():
            if p.type == 'weight':
                p.Vdw = self.beta1 * p.Vdw + ((1 - self.beta1) *  p.value.grad) # momentum
                p.Sdw = self.beta2 * p.Sdw +  ((1 - self.beta2) *  p.value.grad *  p.value.grad )# rmsporp

                Vdw_corrected = p.Vdw / (1 - (self.beta1 ** self.t) )
                Sdw_corected  = p.Sdw  / (1 - (self.beta2 ** self.t ))

                p.value.data += -self.lr * (Vdw_corrected / math.sqrt(Sdw_corected + 1e-8))

            elif p.type == 'bias':
                p.Vdb = self.beta1 * p.Vdb + ( (1-self.beta1) *  p.value.grad) # momentum
                p.Sdb = self.beta2 * p.Sdb +  ((1 - self.beta2) *  p.value.grad) # rmsporp
                Vdb_corrected = p.Vdb / (1 - (self.beta1 ** self.t ))
                Sdb_corrected  = p.Sdb  / (1 - (self.beta2 ** self.t ))
                p.value.data += - self.lr *  (Vdb_corrected / math.sqrt(Sdb_corrected + 1e-8 ))
        

