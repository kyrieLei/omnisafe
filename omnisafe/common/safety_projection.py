import numpy as np
import scipy.signal

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


def discount_cumsum( x, discount):
    """
    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    output=scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
    return output[0]

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation=nn.Tanh, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])




class C_Critic(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.ReLU):
        super().__init__()
        self.c_net = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation, output_activation=nn.Softplus)


    def forward(self, obs_act):
        return torch.squeeze(self.c_net(obs_act), -1)

    # Get the corrected action, this may cause some problems in discrete observation space
    def safety_correction(self, obs, act, delta=0.):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        act = torch.as_tensor(act, dtype=torch.float32)
        print(obs.shape)
        print(act.shape)
        pred = self.forward(torch.cat((obs, act)))

        act_0 = torch.zeros_like(act)
        act_0.requires_grad_()
        self.c_net.zero_grad()
        pred_0 = self.forward(torch.cat((obs, act_0)))
        pred_0.backward(retain_graph=True)

        if pred.item() <= delta:
            return act.detach().cpu().numpy()
        else:
            G = act_0.grad.cpu().data.numpy()
            G = np.asarray(G, dtype=np.double)
            act_np = act.detach().cpu().numpy()
            act_base = act_np * 0
            gamma = 0.01
            epsilon = (1 - gamma) * abs(np.asarray(delta) - self.Q_init)
            top = np.matmul(np.transpose(G), act_np - act_base) - epsilon
            bot = np.matmul(np.transpose(G), G)
            lam = max(0, top / bot)
            act = act + torch.as_tensor(lam * G, dtype=torch.float32)

            return act.detach().cpu().numpy()






