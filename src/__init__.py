"""
Ebalnet: Adapting Neural Networks for Entropy Balancing

A package for causal effect estimation combining neural networks with entropy balancing.
"""

from .ebal_util import NNEbal, ebal_bin, naive_ebal, DataLoader, IHDPLoader, JOBSLoader
from .baseline_methods import IPW, TARNet, CFRNet_WASS, Dragonnet

__version__ = "1.0.0"
__author__ = "Di Ai, Chaowen Zheng, Yuan Feng, Darren Xu"

__all__ = [
    # Core Ebalnet
    "NNEbal",
    "ebal_bin",
    "naive_ebal",
    
    # Data loaders
    "DataLoader",
    "IHDPLoader", 
    "JOBSLoader",
    
    # Baseline methods
    "IPW",
    "TARNet",
    "CFRNet_WASS",
    "Dragonnet",
]

