

import torch
from torch.optim.optimizer import Optimizer, required
class New_optim(Optimizer):
    def __init__(self, params,lr=required, momentum=0, dampening=0, weight_decay1=0, weight_decay2=0, nesterov=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay1=weight_decay1, weight_decay2=weight_decay2, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(New_optim, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(New_optim, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self,inner, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay1 = group['weight_decay1']
            weight_decay2 = group['weight_decay2']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay1 != 0:
                    d_p.add_(weight_decay1, torch.sign(p.data))
                if weight_decay2 != 0:
                    d_p.add_(weight_decay2, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf
                p.data.add_(-group['lr'], d_p*inner.mean())

        return loss
