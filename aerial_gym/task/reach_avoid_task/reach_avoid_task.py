from aerial_gym.task.base_task import BaseTask
from aerial_gym.sim.sim_builder import SimBuilder
import torch
import numpy as np

from aerial_gym.utils.math import *

from aerial_gym.utils.logging import CustomLogger

from aerial_gym.utils.vae.vae_image_encoder import VAEImageEncoder

import gymnasium as gym
from gym.spaces import Dict, Box

logger = CustomLogger("reach_avoid_task")


def dict_to_class(dict):
    return type("ClassFromDict", (object,), dict)


class ReachAvoidTask(BaseTask):
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
        logger.info("Building environment for reach-avoid task.")
        logger.info(
            "Sim Name: {}, Env Name: {}, Robot Name: {}, Controller Name: {}".format(
                self.task_config.sim_name,
                self.task_config.env_name,
                self.task_config.robot_name,
                self.task_config.controller_name,
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

        self.target_position = torch.zeros(
            (self.sim_env.num_envs, 3), device=self.device, requires_grad=False
        )

        self.target_min_ratio = torch.tensor(
            self.task_config.target_min_ratio, device=self.device, requires_grad=False
        ).expand(self.sim_env.num_envs, -1)
        self.target_max_ratio = torch.tensor(
            self.task_config.target_max_ratio, device=self.device, requires_grad=False
        ).expand(self.sim_env.num_envs, -1)

        self.success_aggregate = 0
        self.crashes_aggregate = 0
        self.timeouts_aggregate = 0

        self.pos_error_vehicle_frame_prev = torch.zeros_like(self.target_position)
        self.pos_error_vehicle_frame = torch.zeros_like(self.target_position)

        if self.task_config.vae_config.use_vae:
            self.vae_model = VAEImageEncoder(config=self.task_config.vae_config, device=self.device)
            self.image_latents = torch.zeros(
                (self.sim_env.num_envs, self.task_config.vae_config.latent_dims),
                device=self.device,
                requires_grad=False,
            )
        else:
            self.vae_model = lambda x: x

        # Get the dictionary once from the environment and use it to get the observations later.
        # This is to avoid constant returning of data back and forth across functions as the tensors update and can be read in-place.
        self.obs_dict = self.sim_env.get_obs()
        if "curriculum_level" not in self.obs_dict.keys():
            self.curriculum_level = self.task_config.curriculum.min_level
            self.obs_dict["curriculum_level"] = self.curriculum_level
        else:
            self.curriculum_level = self.obs_dict["curriculum_level"]
        self.obs_dict["num_obstacles_in_env"] = self.curriculum_level
        self.curriculum_progress_fraction = (
            self.curriculum_level - self.task_config.curriculum.min_level
        ) / (self.task_config.curriculum.max_level - self.task_config.curriculum.min_level)

        self.terminations = self.obs_dict["crashes"]
        self.truncations = self.obs_dict["truncations"]
        self.rewards = torch.zeros([self.truncations.shape[0],2], device=self.device)

        self.observation_space = Dict(
            {
                "observations": Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.task_config.observation_space_dim,),
                    dtype=np.float32,
                ),
                "image_obs": Box(
                    low=-1.0,
                    high=1.0,
                    shape=(1, 135, 240),
                    dtype=np.float32,
                ),
            }
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.action_transformation_function = self.task_config.action_transformation_function

        self.num_envs = self.sim_env.num_envs

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

        self.num_task_steps = 0

    def close(self):
        self.sim_env.delete_env()

    def reset(self):
        self.reset_idx(torch.arange(self.sim_env.num_envs))
        return self.get_return_tuple()

    def reset_idx(self, env_ids):
        # Sets target --> edit for reach-avoid
        target_ratio = torch_rand_float_tensor(self.target_min_ratio, self.target_max_ratio)
        self.target_position[env_ids] = torch_interpolate_ratio(
            min=self.obs_dict["env_bounds_min"][env_ids],
            max=self.obs_dict["env_bounds_max"][env_ids],
            ratio=target_ratio[env_ids],
        )
        # logger.warning(f"reset envs: {env_ids}")
        self.infos = {}
        return

    def render(self):
        return self.sim_env.render()

    def logging_sanity_check(self, infos):
        # Consider editing since there is no task success
        successes = infos["successes"]
        crashes = infos["crashes"]
        timeouts = infos["timeouts"]
        time_at_crash = torch.where(
            crashes > 0,
            self.sim_env.sim_steps,
            self.task_config.episode_len_steps * torch.ones_like(self.sim_env.sim_steps),
        )
        env_list_for_toc = (time_at_crash < 5).nonzero(as_tuple=False).squeeze(-1)
        crash_envs = crashes.nonzero(as_tuple=False).squeeze(-1)
        success_envs = successes.nonzero(as_tuple=False).squeeze(-1)
        timeout_envs = timeouts.nonzero(as_tuple=False).squeeze(-1)

        if len(env_list_for_toc) > 0:
            logger.critical("Crash is happening too soon.")
            logger.critical(f"Envs crashing too soon: {env_list_for_toc}")
            logger.critical(f"Time at crash: {time_at_crash[env_list_for_toc]}")

        if torch.sum(torch.logical_and(successes, crashes)) > 0:
            logger.critical("Success and crash are occuring at the same time")
            logger.critical(
                f"Number of crashes: {torch.count_nonzero(crashes)}, Crashed envs: {crash_envs}"
            )
            logger.critical(
                f"Number of successes: {torch.count_nonzero(successes)}, Success envs: {success_envs}"
            )
            logger.critical(
                f"Number of common instances: {torch.count_nonzero(torch.logical_and(crashes, successes))}"
            )
        if torch.sum(torch.logical_and(successes, timeouts)) > 0:
            logger.critical("Success and timeout are occuring at the same time")
            logger.critical(
                f"Number of successes: {torch.count_nonzero(successes)}, Success envs: {success_envs}"
            )
            logger.critical(
                f"Number of timeouts: {torch.count_nonzero(timeouts)}, Timeout envs: {timeout_envs}"
            )
            logger.critical(
                f"Number of common instances: {torch.count_nonzero(torch.logical_and(successes, timeouts))}"
            )
        if torch.sum(torch.logical_and(crashes, timeouts)) > 0:
            logger.critical("Crash and timeout are occuring at the same time")
            logger.critical(
                f"Number of crashes: {torch.count_nonzero(crashes)}, Crashed envs: {crash_envs}"
            )
            logger.critical(
                f"Number of timeouts: {torch.count_nonzero(timeouts)}, Timeout envs: {timeout_envs}"
            )
            logger.critical(
                f"Number of common instances: {torch.count_nonzero(torch.logical_and(crashes, timeouts))}"
            )
        return

    def check_and_update_curriculum_level(self, successes, crashes, timeouts):
        self.success_aggregate += torch.sum(successes)
        self.crashes_aggregate += torch.sum(crashes)
        self.timeouts_aggregate += torch.sum(timeouts)

        instances = self.success_aggregate + self.crashes_aggregate + self.timeouts_aggregate

        if instances >= self.task_config.curriculum.check_after_log_instances:
            success_rate = self.success_aggregate / instances
            crash_rate = self.crashes_aggregate / instances
            timeout_rate = self.timeouts_aggregate / instances

            if success_rate > self.task_config.curriculum.success_rate_for_increase:
                self.curriculum_level += self.task_config.curriculum.increase_step
            elif success_rate < self.task_config.curriculum.success_rate_for_decrease:
                self.curriculum_level -= self.task_config.curriculum.decrease_step

            # clamp curriculum_level
            self.curriculum_level = min(
                max(self.curriculum_level, self.task_config.curriculum.min_level),
                self.task_config.curriculum.max_level,
            )
            self.obs_dict["curriculum_level"] = self.curriculum_level
            self.obs_dict["num_obstacles_in_env"] = self.curriculum_level
            self.curriculum_progress_fraction = (
                self.curriculum_level - self.task_config.curriculum.min_level
            ) / (self.task_config.curriculum.max_level - self.task_config.curriculum.min_level)

            logger.warning(
                f"Curriculum Level: {self.curriculum_level}, Curriculum progress fraction: {self.curriculum_progress_fraction}"
            )
            logger.warning(
                f"\nSuccess Rate: {success_rate}\nCrash Rate: {crash_rate}\nTimeout Rate: {timeout_rate}"
            )
            logger.warning(
                f"\nSuccesses: {self.success_aggregate}\nCrashes : {self.crashes_aggregate}\nTimeouts: {self.timeouts_aggregate}"
            )
            self.success_aggregate = 0
            self.crashes_aggregate = 0
            self.timeouts_aggregate = 0

    def process_image_observation(self):
        image_obs = self.obs_dict["depth_range_pixels"].squeeze(1)
        if self.task_config.vae_config.use_vae:
            self.image_latents[:] = self.vae_model.encode(image_obs)
        # # comments to make sure the VAE does as expected
        # decoded_image = self.vae_model.decode(self.image_latents[0].unsqueeze(0))
        # image0 = image_obs[0].cpu().numpy()
        # decoded_image0 = decoded_image[0].squeeze(0).cpu().numpy()
        # # save as .png with timestep
        # if not hasattr(self, "img_ctr"):
        #     self.img_ctr = 0
        # self.img_ctr += 1
        # import matplotlib.pyplot as plt
        # plt.imsave(f"image0{self.img_ctr}.png", image0, vmin=0, vmax=1)
        # plt.imsave(f"decoded_image0{self.img_ctr}.png", decoded_image0, vmin=0, vmax=1)

    def step(self, actions):
        # this uses the action, gets observations
        # calculates rewards, returns tuples
        # In this case, the episodes that are terminated need to be
        # first reset, and the first obseration of the new episode
        # needs to be returned.

        transformed_action = self.action_transformation_function(actions)
        logger.debug(f"raw_action: {actions[0]}, transformed action: {transformed_action[0]}")
        self.sim_env.step(actions=transformed_action)

        # This step must be done since the reset is done after the reward is calculated.
        # This enables the robot to send back an updated state, and an updated observation to the RL agent after the reset.
        # This is important for the RL agent to get the correct state after the reset.
        self.rewards[:], self.terminations[:] = self.compute_rewards_and_crashes(self.obs_dict)

        # logger.info(f"Curriculum Level: {self.curriculum_level}")

        if self.task_config.return_state_before_reset == True:
            return_tuple = self.get_return_tuple()

        self.truncations[:] = torch.where(
            self.sim_env.sim_steps > self.task_config.episode_len_steps,
            torch.ones_like(self.truncations),
            torch.zeros_like(self.truncations),
        )

        # edit success logic
        # successes are are the sum of the environments which are to be truncated and have reached the target within a distance threshold
        successes = self.truncations * (
            #torch.norm(self.target_position - self.obs_dict["robot_position"], dim=1) < 1.0
            self.rewards[:,0] > 0
        )
        successes = torch.where(self.terminations > 0, torch.zeros_like(successes), successes)
        timeouts = torch.where(
            self.truncations > 0, torch.logical_not(successes), torch.zeros_like(successes)
        )
        timeouts = torch.where(
            self.terminations > 0, torch.zeros_like(timeouts), timeouts
        )  # timeouts are not counted if there is a crash

        self.infos["successes"] = successes
        self.infos["timeouts"] = timeouts
        self.infos["crashes"] = self.terminations
        # self.infos["safety_margin"] = safety_margin

        self.logging_sanity_check(self.infos)
        self.check_and_update_curriculum_level(
            self.infos["successes"], self.infos["crashes"], self.infos["timeouts"]
        )
        # rendering happens at the post-reward calculation step since the newer measurement is required to be
        # sent to the RL algorithm as an observation and it helps if the camera image is updated then
        reset_envs = self.sim_env.post_reward_calculation_step()
        if len(reset_envs) > 0:
            self.reset_idx(reset_envs)
        self.num_task_steps += 1
        # do stuff with the image observations here
        self.process_image_observation()
        # self.post_image_reward_addition()
        if self.task_config.return_state_before_reset == False:
            return_tuple = self.get_return_tuple()
        return return_tuple

    def post_image_reward_addition(self):
        image_obs = 10.0 * self.obs_dict["depth_range_pixels"].squeeze(1)
        image_obs[image_obs < 0] = 10.0
        self.min_pixel_dist = torch.amin(image_obs, dim=(1, 2))
        self.rewards[self.terminations < 0] += -exponential_reward_function(
            4.0, 1.0, self.min_pixel_dist[self.terminations < 0]
        )

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
        vec_to_tgt = quat_rotate_inverse(
            self.obs_dict["robot_vehicle_orientation"],
            (self.target_position - self.obs_dict["robot_position"]),
        )
        perturbed_vec_to_tgt = vec_to_tgt + 0.1 * 2 * (torch.rand_like(vec_to_tgt - 0.5))
        dist_to_tgt = torch.norm(vec_to_tgt, dim=-1)
        perturbed_unit_vec_to_tgt = perturbed_vec_to_tgt / dist_to_tgt.unsqueeze(1)
        self.task_obs["observations"][:, 0:3] = perturbed_unit_vec_to_tgt
        self.task_obs["observations"][:, 3] = dist_to_tgt
        # self.task_obs["observation"][:, 3] = self.infos["successes"]
        # self.task_obs["observations"][:, 3:7] = self.obs_dict["robot_vehicle_orientation"]
        euler_angles = ssa(self.obs_dict["robot_euler_angles"])
        perturbed_euler_angles = euler_angles + 0.1 * (torch.rand_like(euler_angles) - 0.5)
        self.task_obs["observations"][:, 4] = perturbed_euler_angles[:, 0]
        self.task_obs["observations"][:, 5] = perturbed_euler_angles[:, 1]
        self.task_obs["observations"][:, 6] = 0.0
        self.task_obs["observations"][:, 7:10] = self.obs_dict["robot_body_linvel"]
        self.task_obs["observations"][:, 10:13] = self.obs_dict["robot_body_angvel"]
        self.task_obs["observations"][:, 13:17] = self.obs_dict["robot_actions"]
        if self.task_config.vae_config.use_vae:
            self.task_obs["observations"][:, 17:] = self.image_latents
        self.task_obs["rewards"] = self.rewards
        self.task_obs["terminations"] = self.terminations
        self.task_obs["truncations"] = self.truncations

        self.task_obs["image_obs"] = self.obs_dict["depth_range_pixels"]

    def compute_rewards_and_crashes(self, obs_dict):
        # edit reward evaluation
        robot_position = obs_dict["robot_position"]
        robot_linvel = obs_dict["robot_body_linvel"]
        angular_velocity = obs_dict["robot_body_angvel"]
        euler_angles = obs_dict["robot_euler_angles"]
        image_obs = obs_dict["depth_range_pixels"]
        return compute_reward(
            robot_position,
            robot_linvel,
            angular_velocity,
            euler_angles,
            image_obs,
            obs_dict["crashes"],
            self.curriculum_progress_fraction,
            self.task_config.reward_parameters,
        )


@torch.jit.script
def exponential_reward_function(
    magnitude: float, exponent: float, value: torch.Tensor
) -> torch.Tensor:
    """Exponential reward function"""
    return magnitude * torch.exp(-(value * value) * exponent)


@torch.jit.script
def exponential_penalty_function(
    magnitude: float, exponent: float, value: torch.Tensor
) -> torch.Tensor:
    """Exponential reward function"""
    return magnitude * (torch.exp(-(value * value) * exponent) - 1.0)


@torch.jit.script
def compute_reward(
    robot_pos,
    robot_linvel,
    angular_velocity,
    euler_angles,
    image_obs,
    crashes,
    curriculum_progress_fraction,
    parameter_dict
):
    min_wall_dist = torch.min(torch.cat([robot_pos, 1 - robot_pos], 1), 1)[0]
    min_obs_dist = torch.amin(image_obs, dim=(2, 3)).squeeze(1)
    
    velocity_max = torch.tensor(parameter_dict["velocity_max"], device=robot_linvel.device)
    angvel_max = torch.tensor(parameter_dict["angvel_max"], device=angular_velocity.device)
    angle_max = torch.tensor(parameter_dict["angle_max"], device=euler_angles.device)
    obs_lmin = torch.tensor(parameter_dict["obs_dist_lmin"], device=robot_linvel.device)
    wall_lmin = torch.tensor(parameter_dict["wall_dist_lmin"], device=min_wall_dist.device)


    obs_gmin = torch.tensor(parameter_dict["obs_dist_gmin"], device=robot_linvel.device)
    wall_gmin = torch.tensor(parameter_dict["wall_dist_gmin"], device=min_wall_dist.device)

    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Dict[str, Tensor]) -> Tuple[Tensor, Tensor] 
    # edit reward (l(x)) computation
    # vel_diff = velocity_max - torch.abs(robot_linvel)
    # angvel_diff = angvel_max - torch.abs(angular_velocity)
    # angle_diff = angle_max - torch.abs(euler_angles[:, ])
    # vel_min = torch.min(vel_diff, dim=1)[0]

    # reward = torch.min(torch.cat(((velocity_max - torch.abs(robot_linvel[:, 0])).unsqueeze(1),
    #                                (velocity_max - torch.abs(robot_linvel[:, 1])).unsqueeze(1),
    #                                (velocity_max - torch.abs(robot_linvel[:, 2])).unsqueeze(1),
    #                                (angvel_max - torch.abs(angular_velocity[:, 0])).unsqueeze(1),
    #                                (angvel_max - torch.abs(angular_velocity[:, 1])).unsqueeze(1),
    #                                (angvel_max - torch.abs(angular_velocity[:, 2])).unsqueeze(1),
    #                                (angle_max - torch.abs(euler_angles[:, 0])).unsqueeze(1),
    #                                (angle_max - torch.abs(euler_angles[:, 1])).unsqueeze(1),
    #                                (obs_lmin - torch.amin(image_obs, [1,2])).unsqueeze(1),
    #                                (wall_lmin - min_wall_dist).unsqueeze(1)), 1), 1)[0]
    
    # print("Robot linear velocity:",robot_linvel)
    # print("Robot angular velocity:",angular_velocity)
    # print("Robot angles:",euler_angles)
    # print("Obstacle distance:",min_obs_dist)
    # print("Position:", robot_pos)
    # print("Wall distance:", min_wall_dist)

    abs_metrics = torch.stack([velocity_max - torch.abs(robot_linvel[:, 0]),
                                velocity_max - torch.abs(robot_linvel[:, 1]),
                                velocity_max - torch.abs(robot_linvel[:, 2]),
                                angvel_max - torch.abs(angular_velocity[:, 0]),
                                angvel_max - torch.abs(angular_velocity[:, 1]),
                                angvel_max - torch.abs(angular_velocity[:, 2]),
                                angle_max - torch.abs(euler_angles[:, 0]),
                                angle_max - torch.abs(euler_angles[:, 1]),
                                min_obs_dist - obs_lmin #, min_wall_dist - wall_lmin
                              ], dim=1)  # shape: [num_envs, 10]
    reward = torch.min(abs_metrics, dim=1)[0]
    
    # g_x = torch.min(torch.cat(((obs_gmin - torch.amin(image_obs, [1,2])).unsqueeze(1),
    #                             (wall_gmin - min_wall_dist).unsqueeze(1)), 1), 1)[0]

    g_x_metrics = torch.stack([min_obs_dist - obs_gmin #,  # shape: [num_envs]
                                #min_wall_dist - wall_gmin                     # shape: [num_envs]
                              ], dim=1)  # shape: [num_envs, 2]
    g_x = torch.min(g_x_metrics, dim=1)[0]  # shape: [num_envs]


    reward[:] = torch.where(
        crashes > 0,
        parameter_dict["collision_penalty"] * torch.ones_like(reward),
        reward,
    )
    return torch.stack([reward, g_x], dim=1), crashes
