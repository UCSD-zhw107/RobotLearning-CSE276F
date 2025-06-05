from dataclasses import dataclass
from typing import Sequence
import numpy as np

@dataclass
class TestEnvConfig:

    # scene config
    table_height: float = 1e-3
    agent_p: Sequence[float] = (-0.1, 1.0, 0)
    agent_q: Sequence[float] = (0.7071, 0, 0, -0.7072)

    # cube config
    cube_halfsize: float = 0.02
    cube_mass: float = 0.08
    

    # target config
    goal_radius: float = 0.10
    goal_distance: float = 0.1
    

    # camera config
    eye: Sequence[float] = (-0.6, 1.3, 0.8)
    target: Sequence[float] = (0.0, 0.13, 0.0)
    

    # bin confg
    bin_wall_halfsize: Sequence[float] = (0.08, 0.008, 0.02)
    bin_bottom_halfsize: Sequence[float] = (0.08, 0.08, 0.008)
    grasp_point_radius: float = 0.01
    
