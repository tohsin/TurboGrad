from turbograd.engine_ import Value
from turbograd.nn import Layer,  Neuron, Module
from turbograd.functional import detach, MSE
import random
import gym

class Critic(Module):
    def __init__(self, nin, nouts) -> None:
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1],'relu') for i in range(len(nouts)-1)]
        self.layers.append(Layer(sz[-2], sz[-1], ''))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
class Actor(Module):
    def __init__(self, nin, nouts, act_limit) -> None:
        self.act_limit = act_limit
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1],'relu') for i in range(len(nouts)-1)]
        self.layers.append(Layer(sz[-2], sz[-1], 'tanh'))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.act_limit * x 

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
env = gym.make("CartPole-v1")
action_space = env.action_space
observation_space = env.action_space
epsilon = 0.4
actor = Agent(4, [32, 32, action_space.n])
epochs = 100
batch_size = 200
from collections import deque
losses = []
buffer = deque(maxlen=1000)
h = 50 # max number of moves
gamma = 0.99

for i in range(epochs):
    done = False
    obs = env.reset()
    obs = obs[0].tolist()
    iter = 0
    while not done or iter< h:
        q_values = actor(obs)
        
        # action selection in epsilon greedy
        pr = random.random()
        if (pr < epsilon):
            action = random.randint(0,1) 
        else:
            if q_values[0].data> q_values[1].data:
                action = 0
            else:
                action = 1
       
        new_obs, reward, done,_, info = env.step(action)
        new_obs = new_obs.tolist()
        mdp = (obs, action, reward, new_obs, done)
        buffer.append(mdp)
        obs = new_obs
        h+1

        if len(buffer)> batch_size:
            minibatch = random.sample(buffer, batch_size)
            s_batch = [s1 for (s1, a, r, s2, d ) in minibatch]
            a_batch = [a for (s1, a, r, s2, d ) in minibatch]
            r_batch = [r for (s1, a, r, s2, d ) in minibatch]
            s2_batch = [s2 for (s1, a, r, s2, d ) in minibatch]
            d_batch = [d for (s1, a, r, s2, d ) in minibatch]

            Q_s = []
            Q_s_1 = []
            target_buff = []
            for i in range(len(s_batch)):
                Q_s.append(actor(s_batch[i])[a_batch[i]])
            ##  you need to not have the Q values with diffrentiable parameters
            for state2 in s2_batch:
                Q_s_1.append(detach(actor(state2)))
        
            target = [reward  +  (gamma * ((1 - d)* max(Q_2))) for reward, d, Q_2 in zip(r_batch, d_batch, Q_s_1 )]
            loss : Value = MSE(Q_s , target)
          
            actor.zero_grad()
            loss.backward()
            actor.step(lr = 0.0001)
            losses.append(loss.data)
            print(loss.data)
            
            






           
            

