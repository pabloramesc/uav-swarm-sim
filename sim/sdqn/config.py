from dataclasses import dataclass
from typing import Type

from .frame_generators import FrameGenerator, GridFrameGenerator

@dataclass
class FrameGeneratorConfig:
    cls: Type[FrameGenerator]
    num_channels: int
    channel_shape: tuple[int, int]
    
@dataclass
class RewardManagerConfig:
    pass
    
@dataclass
class SDQNConfig:
    frame_generator_config: FrameGeneratorConfig