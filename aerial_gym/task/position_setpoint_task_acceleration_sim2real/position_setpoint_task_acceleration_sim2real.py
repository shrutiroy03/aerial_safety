from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
import torch
import numpy as np

from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

import gymnasium as gym
from gym.spaces import Dict, Box

logger = CustomLogger("position_setpoint_task")


def dict_to_class(dict):
    return type("ClassFromDict", (object,), dict)


class PositionSetpointTaskAccelerationSim2Real(BaseTask):
    def __init__(
        self, task_config, seed=None, num_envs=None, headless=None, device=None, use_warp=None
    ):
        # overwrite the params if user has provided them
        if seed is not None:
            task_config.seed = seed
        if num_envs is not None:
            task_config.num_envs = num_envs
        if headless is not None:
            task_config.headless = headless
        if device is not None:
            task_config.device = device
        if use_warp is not None:
            task_config.use_warp = use_warp

        super().__init__(task_config)
        self.device = self.task_config.device
        # set the each of the elements of reward parameter to a torch tensor
        for key in self.task_config.reward_parameters.keys():
            self.task_config.reward_parameters[key] = torch.tensor(
                self.task_config.reward_parameters[key], device=self.device
            )
        logger.info("Building environment for position setpoint task.")
        logger.info(
            "\nSim Name: {},\nEnv Name: {},\nRobot Name: {}, \nController Name: {}".format(
                self.task_config.sim_name,
                self.task_config.env_name,
                self.task_config.robot_name,
                self.task_config.controller_name,
            )
        )
        logger.info(
            "\nNum Envs: {},\nUse Warp: {},\nHeadless: {}".format(
                self.task_config.num_envs,
                self.task_config.use_warp,
                self.task_config.headless,
            )
        )

        self.sim_env = SimBuilder().build_env(
            sim_name=self.task_config.sim_name,
            env_name=self.task_config.env_name,
            robot_name=self.task_config.robot_name,
            controller_name=self.task_config.controller_name,
            args=self.task_config.args,
            device=self.device,
            num_envs=self.task_config.num_envs,
            use_warp=self.task_config.use_warp,
            headless=self.task_config.headless,
        )

        self.actions = torch.zeros(
            (self.sim_env.num_envs, self.task_config.action_space_dim),
            device=self.device,
            requires_grad=False,
        )
        self.prev_actions = torch.zeros_like(self.actions)

        self.prev_actions_vehicle_frame = torch.zeros_like(self.actions)
        self.actions_vehicle_frame = torch.zeros_like(self.actions)

        self.prev_dist = torch.zeros((self.sim_env.num_envs), device=self.device)
        self.counter = 0

        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )

        # self.action_file = open("actions.txt", "w")

        # Get the dictionary once from the environment and use it to get the observations later.
        # This is to avoid constant retuning of data back anf forth across functions as the tensors update and can be read in-place.
        self.obs_dict = self.sim_env.get_obs()
        self.obs_dict["num_obstacles_in_env"] = 1
        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros(self.truncations.shape[0], device=self.device)

        self.observation_space = Dict(
            {"observations": Box(low=-1.0, high=1.0, shape=(13,), dtype=np.float32)}
        )
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(self.task_config.action_space_dim,),
            dtype=np.float32,
        )
        # self.action_transformation_function = self.sim_env.robot_manager.robot.action_transformation_function

        self.num_envs = self.sim_env.num_envs

        self.counter = 0

        # Currently only the "observations" are sent to the actor and critic.
        # The "priviliged_obs" are not handled so far in sample-factory

        self.task_obs = {
            "observations": torch.zeros(
                (self.sim_env.num_envs, self.task_config.observation_space_dim),
                device=self.device,
                requires_grad=False,
            ),
            "priviliged_obs": torch.zeros(
                (
                    self.sim_env.num_envs,
                    self.task_config.privileged_observation_space_dim,
                ),
                device=self.device,
                requires_grad=False,
            ),
            "collisions": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
            "rewards": torch.zeros(
                (self.sim_env.num_envs, 1), device=self.device, requires_grad=False
            ),
        }

    def close(self):
        self.sim_env.delete_env()

    def reset(self):
        self.target_position[:, 0:3] = 0.0  # torch.rand_like(self.target_position) * 10.0
        self.infos = {}
        self.sim_env.reset()
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        self.target_position[:, 0:3] = (
            0.0  # (torch.rand_like(self.target_position[env_ids]) * 10.0)
        )
        self.infos = {}
        self.sim_env.reset_idx(env_ids)
        return

    def render(self):
        return None

    def step(self, actions):
        self.counter += 1
        self.prev_actions[:] = self.actions
        self.prev_actions_vehicle_frame[:, 0:3] = quat_rotate(
            self.obs_dict["robot_orientation"], self.prev_actions[:, 0:3]
        )
        self.prev_actions_vehicle_frame[:, 3] = self.prev_actions[:, 3]
        self.prev_dist[:] = torch.norm(
            self.target_position - self.obs_dict["robot_position"], dim=1
        )
        self.actions = actions
        self.actions[:, 0:3] = 2.0 * self.actions[:, 0:3]
        # self.action_file.write(f"{self.actions[0].cpu().numpy()[0]}, {self.actions[0].cpu().numpy()[1]}, {self.actions[0].cpu().numpy()[2]}, {self.actions[0].cpu().numpy()[3]},\n")
        # print(self.actions[0].cpu().numpy())

        # this uses the action, gets observations
        # calculates rewards, returns tuples
        # In this case, the episodes that are terminated need to be
        # first reset, and the first obseration of the new episode
        # needs to be returned.
        self.sim_env.step(actions=self.actions)

        # This step must be done since the reset is done after the reward is calculated.
        # This enables the robot to send back an updated state, and an updated observation to the RL agent after the reset.
        # This is important for the RL agent to get the correct state after the reset.
        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)

        if self.task_config.return_state_before_reset == True:
            return_tuple = self.get_return_tuple()

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps, 1, 0
        )
        self.sim_env.post_reward_calculation_step()

        self.infos = {}  # self.obs_dict["infos"]

        if self.task_config.return_state_before_reset == False:
            return_tuple = self.get_return_tuple()

        return return_tuple

    def get_return_tuple(self):
        self.process_obs_for_task()
        return (
            self.task_obs,
            self.rewards,
            self.terminations,
            self.truncations,
            self.infos,
        )

    def process_obs_for_task(self):
        position_error = self.target_position - self.obs_dict["robot_position"]
        self.obs_dict["robot_orientation"][:] = (
            torch.sign(self.obs_dict["robot_orientation"][:, 3]).unsqueeze(1)
            * self.obs_dict["robot_orientation"]
        )

        euler_angles = ssa(get_euler_xyz_tensor(self.obs_dict["robot_orientation"]))
        euler_angles_noisy = euler_angles + torch.randn_like(euler_angles) * 0.02

        self.task_obs["observations"][:, 0:3] = (
            position_error + torch.randn_like(position_error) * 0.03
        )
        self.task_obs["observations"][:, 3:7] = quat_from_euler_xyz_tensor(euler_angles_noisy)
        self.task_obs["observations"][:, 7:10] = (
            self.obs_dict["robot_body_linvel"]
            + torch.randn_like(self.obs_dict["robot_body_linvel"]) * 0.02
        )
        self.task_obs["observations"][:, 10:13] = (
            self.obs_dict["robot_body_angvel"]
            + torch.randn_like(self.obs_dict["robot_body_angvel"]) * 0.02
        )
        self.task_obs["observations"][:, 13:] = self.obs_dict["robot_actions"]

        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations

    def compute_rewards_and_crashes(self, obs_dict):
        robot_position = obs_dict["robot_position"]
        robot_body_linvel = obs_dict["robot_body_linvel"]
        target_position = self.target_position
        robot_vehicle_orientation = obs_dict["robot_vehicle_orientation"]
        robot_orientation = obs_dict["robot_orientation"]
        target_orientation = torch.zeros_like(robot_orientation, device=self.device)
        target_orientation[:, 3] = 1.0
        angular_velocity = obs_dict["robot_body_angvel"]
        root_quats = obs_dict["robot_orientation"]

        pos_error_vehicle_frame = quat_apply_inverse(
            robot_orientation, (target_position - robot_position)
        )

        self.actions_vehicle_frame[:, 0:3] = quat_rotate(
            robot_vehicle_orientation, self.actions[:, 0:3]
        )
        self.actions_vehicle_frame[:, 3] = self.actions[:, 3]

        current_yaw = ssa(get_euler_xyz_tensor(robot_orientation))[:, 2]
        yaw_error = 0 - current_yaw

        return compute_reward(
            pos_error_vehicle_frame,
            self.prev_dist,
            yaw_error,
            robot_body_linvel,
            angular_velocity,
            obs_dict["crashes"],
            1.0,  # obs_dict["curriculum_level_multiplier"],
            self.actions_vehicle_frame,
            self.prev_actions_vehicle_frame,
            self.task_config.reward_parameters,
        )


@torch.jit.script
def exp_func(x, gain, exp):
    # type: (Tensor, float, float) -> Tensor
    return gain * torch.exp(-exp * x * x)


@torch.jit.script
def abs_exp_func(x, gain, exp):
    # type: (Tensor, float, float) -> Tensor
    return gain * torch.exp(-exp * torch.abs(x))


@torch.jit.script
def exp_penalty_func(x, gain, exp):
    # type: (Tensor, float, float) -> Tensor
    return gain * (torch.exp(-exp * x * x) - 1)


@torch.jit.script
def abs_exp_penalty_func(x, gain, exp):
    # type: (Tensor, float, float) -> Tensor
    return gain * (torch.exp(-exp * torch.abs(x)) - 1)


@torch.jit.script
def compute_reward(
    pos_error,
    prev_dist,
    yaw_error,
    robot_linvels,
    robot_angvels,
    crashes,
    curriculum_level_multiplier,
    current_action,
    prev_actions,
    parameter_dict,
):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, Dict[str, Tensor]) -> Tuple[Tensor, Tensor]

    dist = torch.norm(pos_error, dim=1)

    pos_reward = (
        exp_func(dist, 2.0, 1.0) + exp_func(dist, 3.0, 10.0) + abs_exp_func(dist, 3.0, 50.0)
    )

    close_pos_reward = exp_func(dist, 2.0, 1.0)

    robot_speed = torch.norm(robot_linvels, dim=1)

    speed_reward = exp_func(robot_speed, 2.0, 2.5)

    action_penalty = torch.sum(abs_exp_penalty_func(current_action, 0.3, 4.0), dim=1)
    action_difference = current_action - prev_actions
    action_difference_penalty = torch.sum(abs_exp_penalty_func(action_difference, 0.4, 6.0), dim=1)

    closer_reward = torch.where(
        dist < prev_dist, 400.0 * (prev_dist - dist), 1200 * (prev_dist - dist)
    )

    yaw_error_reward = abs_exp_func(yaw_error, 3.0, 5.0)

    total_reward = (
        (
            pos_reward
            + pos_reward * (closer_reward / 9.0 + action_penalty / 3.0 + speed_reward / 1.5)
        )
        + action_penalty
        + action_difference_penalty
        + closer_reward
        + yaw_error_reward
        + close_pos_reward
        + speed_reward * 0.2
    )

    total_reward[:] = curriculum_level_multiplier * total_reward

    crashes[:] = torch.where(dist > 10.0, torch.ones_like(crashes), crashes)

    total_reward[:] = torch.where(crashes > 0.0, -50 * torch.ones_like(total_reward), total_reward)

    return total_reward, crashes
