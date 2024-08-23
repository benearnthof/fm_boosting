"""
Base class wrappers for flow models, full models are implemented in their respective files.
Nice tutorial can be found here:
https://colab.research.google.com/drive/1g8Fm_S4BqrDaG2eHI3sDulBD3WbK2D_V?usp=sharing
https://www.cs.utexas.edu/~lqiang/rectflow/html/intro.html
https://colab.research.google.com/drive/1LouqFBIC7pnubCOl5fhnFd33-oVJao2J?usp=sharing
"""

import torch
from torch.nn import Module
from torch import Tensor

class FlowModel(Module):
    """
    Base Class Wrapper for all Flow Models.
    Classes inheriting from this need to implement their own sample and forward methods.
    """
    def __init__(
        self,
        model: Module,
    ):
        super().__init__()
        self.model = model

    
    @torch.no_grad()
    def sample(
        self,
        z0=None,
        num_steps=1000,
    ):
        raise NotImplementedError

    def forward(
        self,
        data: Tensor,
        noise: Tensor | None = None,
    ):
        raise NotImplementedError
