import torch
from aerial_gym import AERIAL_GYM_DIRECTORY


class task_config:
    seed = -1
    sim_name = "base_sim"
    env_name = "env_with_obstacles"
    robot_name = "lmf2"
    controller_name = "lmf2_velocity_control"
    args = {}
    num_envs = 128
    use_warp = True
    headless = True
    device = "cuda:0"
    observation_space_dim = 13 + 4 + 64  # root_state + action_dim _+ latent_dims
    privileged_observation_space_dim = 0
    action_space_dim = 4
    episode_len_steps = 100  # real physics time for simulation is this value multiplied by sim.dt

    return_state_before_reset = (
        False  # False as usually state is returned for next episode after reset
    )
    # user can set the above to true if they so desire

    target_min_ratio = [0.90, 0.1, 0.1]  # target ratio w.r.t environment bounds in x,y,z
    target_max_ratio = [0.94, 0.90, 0.90]  # target ratio w.r.t environment bounds in x,y,z

    reward_parameters = {
        "velocity_max": 0.2,
        "angvel_max": 0.2,
        "angle_max": 0.4,
        "obs_dist_lmin": 0.1,
        "obs_dist_gmin": 0.05,
        "wall_dist_lmin": 0.1,
        "wall_dist_gmin": 0.05,
        "collision_penalty": -100.0,
    }

    class vae_config:
        use_vae = False
        latent_dims = 64
        model_file = (
            AERIAL_GYM_DIRECTORY
            + "/aerial_gym/utils/vae/weights/ICRA_test_set_more_sim_data_kld_beta_3_LD_64_epoch_49.pth"
        )
        model_folder = AERIAL_GYM_DIRECTORY
        image_res = (270, 480)
        interpolation_mode = "nearest"
        return_sampled_latent = True

    class curriculum:
        min_level = 15
        max_level = 50
        check_after_log_instances = 2048
        increase_step = 2
        decrease_step = 1
        success_rate_for_increase = 0.7
        success_rate_for_decrease = 0.6

        def update_curriculim_level(self, success_rate, current_level):
            if success_rate > self.success_rate_for_increase:
                return min(current_level + self.increase_step, self.max_level)
            elif success_rate < self.success_rate_for_decrease:
                return max(current_level - self.decrease_step, self.min_level)
            return current_level

    # def action_transformation_function(action):
    #     clamped_action = torch.clamp(action, -1.0, 1.0)
    #     max_speed = 1.5  # [m/s]
    #     max_yawrate = torch.pi / 3  # [rad/s]
    #     processed_action = clamped_action.clone()
    #     processed_action[:, 0:3] = max_speed*processed_action[:, 0:3]
    #     processed_action[:, 3] = max_yawrate*processed_action[:, 3]
    #     return processed_action

    def action_transformation_function(action):
        clamped_action = torch.clamp(action, -1.0, 1.0)
        max_speed = 2.0  # [m/s]
        max_yawrate = torch.pi / 3  # [rad/s]

        # clamped_action[:, 0:3] = max_speed * clamped_action[:, 0:3]
        # clamped_action[:, 3] = max_yawrate * clamped_action[:, 3]
        # return clamped_action

        max_inclination_angle = torch.pi / 4  # [rad]

        clamped_action[:, 0] += 1.0

        processed_action = torch.zeros(
            (clamped_action.shape[0], 4), device=task_config.device, requires_grad=False
        )
        processed_action[:, 0] = (
            clamped_action[:, 0]
            * torch.cos(max_inclination_angle * clamped_action[:, 1])
            * max_speed
            / 2.0
        )
        processed_action[:, 1] = 0
        processed_action[:, 2] = (
            clamped_action[:, 0]
            * torch.sin(max_inclination_angle * clamped_action[:, 1])
            * max_speed
            / 2.0
        )
        processed_action[:, 3] = clamped_action[:, 2] * max_yawrate
        return processed_action
