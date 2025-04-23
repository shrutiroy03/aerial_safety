# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_action_isaacgympy
import os
import random
import time

import gym
import isaacgym  # noqa
from isaacgym import gymutil
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


from aerial_gym.registry.task_registry import task_registry
from aerial_gym.utils.helpers import parse_arguments


def get_args():
    custom_parameters = [
        {
            "name": "--task",
            "type": str,
            "default": "reach_avoid_task",
            "help": "Resume training or start testing from a checkpoint. Overrides config file if provided.",
        },
        {
            "name": "--experiment_name",
            "type": str,
            "default": os.path.basename(__file__).rstrip(".py"),
            "help": "Name of the experiment to run or load. Overrides config file if provided.",
        },
        {
            "name": "--checkpoint",
            "type": str,
            "default": None,
            "help": "Saved model checkpoint number.",
        },
        {
            "name": "--headless",
            "action": "store_true",
            "default": False,
            "help": "Force display off at all times",
        },
        {
            "name": "--horovod",
            "action": "store_true",
            "default": False,
            "help": "Use horovod for multi-gpu training",
        },
        {
            "name": "--rl_device",
            "type": str,
            "default": "cuda:0",
            "help": "Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
        },
        {
            "name": "--num_envs",
            "type": int,
            "default": 2048,
            "help": "Number of environments to create. Overrides config file if provided.",
        },
        {
            "name": "--seed",
            "type": int,
            "default": 1,
            "help": "Random seed. Overrides config file if provided.",
        },
        {
            "name": "--play",
            "required": False,
            "help": "only run network",
            "action": "store_true",
        },
        {
            "name": "--torch-deterministic-off",
            "action": "store_true",
            "default": False,
            "help": "if toggled, `torch.backends.cudnn.deterministic=False`",
        },
        {
            "name": "--track",
            "action": "store_true",
            "default": False,
            "help": "if toggled, this experiment will be tracked with Weights and Biases",
        },
        {
            "name": "--wandb-project-name",
            "type": str,
            "default": "cleanRL",
            "help": "the wandb's project name",
        },
        {
            "name": "--wandb-entity",
            "type": str,
            "default": None,
            "help": "the entity (team) of wandb's project",
        },
        # Algorithm specific arguments
        {
            "name": "--total-timesteps",
            "type": int,
            "default": 3000000000,
            "help": "total timesteps of the experiments",
        },
        {
            "name": "--actor-lr",
            "type": float,
            "default": 0.0026,
            "help": "the learning rate of the actor",
        },
        {
            "name": "--critic-lr",
            "type": float,
            "default": 0.0026,
            "help": "the learning rate of the critics",
        },
        {
            "name": "--num-steps",
            "type": int,
            "default": 32,
            "help": "the number of steps to run in each environment per policy rollout",
        },
        {
            "name": "--start-steps",
            "type": int,
            "default": 10000,
            "help": "the number of steps for uniform-random action selection before the real policy",
        },
        {
            "name": "--anneal-lr",
            "action": "store_true",
            "default": False,
            "help": "Toggle learning rate annealing for policy and value networks",
        },
        {
            "name": "--gamma",
            "type": float,
            "default": 0.99,
            "help": "the discount factor gamma",
        },
        {
            "name": "--alpha",
            "type": float,
            "default": 0.2,
            "help": "the entropy regularization coefficient alpha",
        },
        {
            "name": "--tau",
            "type": float,
            "default": 0.005,
            "help": "the soft update coefficient tau",
        },
        {
            "name": "--replay-size",
            "type": int,
            "default": 1000,
            "help": "the maximum length of the replay buffer",
        },
        {
            "name": "--num-minibatches",
            "type": int,
            "default": 2,
            "help": "the number of mini-batches",
        },
        {
            "name": "--actor-update-epochs",
            "type": int,
            "default": 4,
            "help": "the K epochs to update the actor",
        },
        {
            "name": "--target-update-epochs",
            "type": int,
            "default": 50,
            "help": "the K epochs to update the target networks",
        },
        {
            "name": "--norm-adv-off",
            "action": "store_true",
            "default": False,
            "help": "Toggles advantages normalization",
        },
        {
            "name": "--clip-coef",
            "type": float,
            "default": 0.2,
            "help": "the surrogate clipping coefficient",
        },
        {
            "name": "--clip-vloss",
            "action": "store_true",
            "default": False,
            "help": "Toggles whether or not to use a clipped loss for the value function, as per the paper.",
        },
        {
            "name": "--ent-coef",
            "type": float,
            "default": 0.0,
            "help": "coefficient of the entropy",
        },
        {
            "name": "--vf-coef",
            "type": float,
            "default": 2,
            "help": "coefficient of the value function",
        },
        {
            "name": "--max-grad-norm",
            "type": float,
            "default": 1,
            "help": "the maximum norm for the gradient clipping",
        }
    ]

    # parse arguments
    args = parse_arguments(description="RL Policy", custom_parameters=custom_parameters)

    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    args.torch_deterministic = not args.torch_deterministic_off
    args.norm_adv = not args.norm_adv_off

    # name allignment
    args.sim_device_id = args.compute_device_id
    args.sim_device = args.sim_device_type
    if args.sim_device == "cuda":
        args.sim_device += f":{args.sim_device_id}"
    return args


class RecordEpisodeStatisticsTorch(gym.Wrapper):
    def __init__(self, env, device):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.device = device
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.returned_episode_returns = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.returned_episode_lengths = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        return observations

    def step(self, action):
        observations, rewards, terminations, truncations, infos = super().step(action)

        self.episode_returns += rewards
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        dones = torch.where(terminations | truncations, 1, 0).to(self.device)
        self.episode_returns *= 1 - dones
        self.episode_lengths *= 1 - dones
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )
    
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.sfts_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros([size], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, sft, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.sfts_buf[self.ptr] = sft
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=256):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(
            obs=self.obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            sfts=self.sfts_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            done=self.done_buf[idxs],
        )


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        '''
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.task_config.observation_space_dim).prod(), 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        ) '''
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.task_config.observation_space_dim).prod(), 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, np.prod(envs.task_config.action_space_dim)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.task_config.action_space_dim)))

        self.q1 = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.task_config.observation_space_dim).prod() + np.prod(envs.task_config.action_space_dim), 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1))
        )
        self.q2 = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.task_config.observation_space_dim).prod() + np.prod(envs.task_config.action_space_dim), 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1))
        )

        self.target_q1 = copy.deepcopy(self.q1)
        self.target_q2 = copy.deepcopy(self.q2)

    def get_target_value(self, x, action):
        obs_action = torch.cat([x, action], dim=-1)
        q1 = self.target_q1(obs_action)
        q2 = self.target_q2(obs_action)
        return q1, q2

    def get_value(self, x, action):
        obs_action = torch.cat([x, action], dim=-1)
        q1 = self.q1(obs_action)
        q2 = self.q2(obs_action)
        return q1, q2

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        q1, q2 = self.get_value(x, action)
        return (
            action,
            probs.log_prob(action).sum(1),
            probs.entropy().sum(1),
            q1,
            q2
        )

@torch.no_grad()
def soft_update(target, source, tau):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

if __name__ == "__main__":
    args = get_args()

    run_name = f"{args.task}__{args.experiment_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = args.sim_device
    print("using device:", device)

    # env setup
    envs = task_registry.make_task(task_name=args.task)
    envs = RecordEpisodeStatisticsTorch(envs, device)
    obs_dim = envs.task_config.observation_space_dim
    act_dim = envs.task_config.action_space_dim

    print("num actions: ", envs.task_config.action_space_dim)
    print("num obs: ", envs.task_config.observation_space_dim)

    agent = Agent(envs).to(device)
    actor_optim = optim.Adam(agent.actor_mean.parameters(), lr=args.actor_lr, eps=1e-5)
    q1_optim = optim.Adam(agent.q1.parameters(), lr=args.critic_lr, eps=1e-5)
    q2_optim = optim.Adam(agent.q2.parameters(), lr=args.critic_lr, eps=1e-5)

    buffer = ReplayBuffer(obs_dim, act_dim, size=int(1e6))
    alpha = args.alpha
    gamma = args.gamma
    tau = args.tau

    if args.play and args.checkpoint is None:
        raise ValueError("No checkpoint provided for testing.")

    # load checkpoint if needed
    if args.checkpoint is not None:
        print("Loading checkpoint...")
        checkpoint = torch.load(args.checkpoint)
        agent.load_state_dict(checkpoint)
        print("Loaded checkpoint")

    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs, envs.task_config.observation_space_dim),
        dtype=torch.float,
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs, envs.task_config.action_space_dim),
        dtype=torch.float,
    ).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    safety = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    q1values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    q2values = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)
    targets = torch.zeros((args.num_steps, args.num_envs), dtype=torch.float).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _, _, _, _info = envs.reset()
    next_obs = next_obs["observations"]
    next_done = torch.zeros(args.num_envs, dtype=torch.float).to(device)
    num_updates = args.total_timesteps // args.batch_size

    if not args.play:
        for update in range(1, num_updates + 1):
            # Annealing the rate if instructed to do so.
            '''if args.anneal_lr:
                frac = 1.0 - (update - 1.0) / num_updates
                lrnow = frac * args.actor_lr
                actor_optim.param_groups[0]["lr"] = lrnow'''

            for step in range(0, args.num_steps):
                global_step += 1 * args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                '''if global_step < args.start_steps:
                    action = np.random.uniform(-1, 1, size=(envs.num_envs, act_dim))
                    logprobs[step] = logprob
                else:'''
                with torch.no_grad():
                    action, logprob, _, q1, q2 = agent.get_action_and_value(next_obs)
                    q1values[step] = q1.flatten()
                    q2values[step] = q2.flatten()
                logprobs[step] = logprob
                actions[step] = torch.tensor(action, dtype=torch.float32, device=device)

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, rewards[step], next_done, info = envs.step(action)
                safety[step] = info["safety_margin"]
                next_obs = next_obs["observations"]
                if 0 <= step <= 2:
                    for idx, d in enumerate(next_done):
                        if d:
                            episodic_return = info["r"][idx].item()
                            print(f"global_step={global_step}, episodic_return={episodic_return}")
                            writer.add_scalar(
                                "charts/episodic_return", episodic_return, global_step
                            )
                            writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)
                            if "consecutive_successes" in info:  # ShadowHand and AllegroHand metric
                                writer.add_scalar(
                                    "charts/consecutive_successes",
                                    info["consecutive_successes"].item(),
                                    global_step,
                                )
                            break
                
                buffer.store(obs[step], action, rewards[step], safety[step], next_obs, next_done)
            
            # bootstrap value if not done
            with torch.no_grad():
                next_q1, next_q2 = agent.get_target_value(next_obs, action)
                next_q1 = next_q1.reshape(1, -1)
                next_q2 = next_q2.reshape(1, -1)
                target_V = torch.min(next_q1, next_q2)
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = target_V
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = targets[t + 1]
                    target_Q = (1.0 - args.gamma) * torch.max(rewards[t], safety[t]) + args.gamma * torch.max(torch.min(rewards[t], nextvalues * nextnonterminal), safety[t])
                    targets[t] = target_Q

            # flatten the batch
            b_obs = obs.reshape((-1, envs.task_config.observation_space_dim))
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1, envs.task_config.action_space_dim))
            b_targets = targets.reshape(-1)
            b_q1values = q1values.reshape(-1)
            b_q2values = q2values.reshape(-1)

            # Optimizing the policy and value network
            clipfracs = []
            for epoch in range(args.update_epochs):
                b_inds = torch.randperm(args.batch_size, device=device)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newq1, newq2 = agent.get_action_and_value(
                        b_obs[mb_inds], b_actions[mb_inds]
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    '''with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
                    '''

                    # Critic loss
                    c_loss = ((newq1 - b_targets) ** 2).mean() + ((newq2 - b_targets) ** 2).mean()
                    q1_optim.zero_grad()
                    q2_optim.zero_grad()
                    c_loss.backward()
                    q1_optim.step()
                    q2_optim.step()

                    if epoch % args.actor_update_epochs == 0:
                        # Actor loss
                        a_loss = (alpha * newlogprob - torch.min(newq1, newq2)).mean()
                        actor_optim.zero_grad()
                        a_loss.backward()
                        nn.utils.clip_grad_norm_(agent.actor_mean.parameters(), args.max_grad_norm)
                        actor_optim.step()

                    if epoch % args.target_update_epochs == 0:
                        soft_update(agent.target_q1, agent.q1, tau)
                        soft_update(agent.target_q2, agent.q2, tau)

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/actor_loss", a_loss.item(), global_step)
            writer.add_scalar("losses/critic_loss", c_loss.item(), global_step)
            # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            # writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar(
                "charts/SPS", int(global_step / (time.time() - start_time)), global_step
            )

            # save the model levery 50 updates
            if update % 50 == 0:
                print("Saving model.")
                torch.save(agent.state_dict(), f"runs/{run_name}/latest_model.pth")

    else:
        for step in range(0, 5000000):
            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, _, _ = agent.get_action_and_value(next_obs)
            next_obs, rewards, next_done, info = envs.step(action)

    # envs.close()
    writer.close()
