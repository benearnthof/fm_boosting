# outline of rectified flow
# https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html#method-rectified-flow
# still missing reflow but may not be needed will investigate tomorrow

import torch
from torch.nn import Module
from flow_models import FlowModel

class RectifiedFlow(FlowModel):
    """
    Basic Rectified Flow Model with data coupling.
    """
    def __init__(
        self,
        model=Module,
        ):
        super().__init__(model=model)
        
    def get_train_tuple(self, x0=None, x1=None)
        t = torch.rand((x1.shape[0], 1)) # batch size dictates number of time samples
        x_t = t * x1 + (1. -t) * x0 # linear interpolation, TODO: add eps value to stabilize training
        target = x1 - x0
        return x_t, t, target
    
    @torch.no_grad()
    def sample(self, x0=None, steps=25):
        """
        Simple Euler ODE sampling
        """
        dt = 1/steps

        trajectory = []
        x = x0.clone()
        batch_size, *data_shape = x.shape

        trajectory.append(x.clone())
        for i in range(steps):
            t = torch.ones((batch_size, 1)) * i / steps
            pred = self.model(x, t)
            x = x.clone() + pred *dt
            trajectory.append(x.clone()) # this seems very inefficient, we should do this with torchdiffeq

        return trajectory

    def forward(self, x0):
        raise NotImplementedError


