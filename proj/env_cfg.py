from dataclasses import dataclass
from typing import Sequence
import numpy as np

@dataclass
class EnvConfig:

    # scene config
    table_height: float = 1e-3
    agent_p: Sequence[float] = (-0.1, 1.0, 0)
    agent_q: Sequence[float] = (0.7071, 0, 0, -0.7072)

    # cube config
    cube_halfsize: float = 0.02
    cube_mass: float = 0.08
    

    # target config
    goal_radius: float = 0.25
    goal_distance: float = 0.5
    

    # camera config
    eye: Sequence[float] = (-0.6, 1.3, 0.8)
    target: Sequence[float] = (0.0, 0.13, 0.0)
    

    # evaluation config
    throw_air_threshold: float = 0.1
    throw_vel_xy_threshold: float = 0.5
    throw_vel_threshold: float = 2.0



@dataclass
class RewardConfig:
    # reward config
    approach_reward_weight: float = 2.0
    grasp_reward_weight: float = 2.0
    lift_reward_weight: float = 2.0
    direction_align_reward_weight: float = 5.0
    throw_reward_weight: float = 3.0
    success_reward_weight: float = 20.0

    # penalty config
    drop_penalty_weight: float = -2.0
    action_penalty_weight: float = -0.01


    # reward config
    target_lift_height: float = 0.1