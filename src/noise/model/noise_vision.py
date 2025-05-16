import numpy as np
import torch
import torch.nn as nn
from noise import NoiseModel

class VisionNoiseModel(NoiseModel, nn.Module):
    def __init__(self, config):
        NoiseModel.__init__(self, config)
        nn.Module.__init__(self)
