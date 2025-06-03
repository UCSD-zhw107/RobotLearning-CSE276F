import sapien
import numpy as np
import gymnasium as gym
from mani_skill.utils import sapien_utils, common
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.wrappers.record import RecordEpisode
import torch
from transforms3d.euler import euler2quat
from env_cfg import EnvConfig
from mani_skill.utils.building import actors
from env import ThrowCubePandasEnv

"""
This script is used to test the demo agents with random actions.
"""

def main():
    env = gym.make("ThrowCubePandas-v1",
                num_envs=1,
                control_mode="pd_joint_delta_pos",
                render_mode="rgb_array",
                reward_mode="none",
                human_render_camera_configs=dict(shader_pack="default")
    )
    env = RecordEpisode(env, output_dir="random_agents", video_fps=20, info_on_video=False)
    env.reset(seed=42)
    for t in range(50):
        _, _, _, _, info = env.step(env.action_space.sample())
    img = env.render().cpu().numpy()[0]
    print(f'Video saved to /random_agents')
    env.close()


if __name__ == "__main__":
    main()