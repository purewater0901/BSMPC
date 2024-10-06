import envs.dm_control as dm_control
from utils.replay_buffer import ReplayBuffer, convert_dict
from utils.config_parser import parse_cfg
from agent.tdmpc import TDMPC
from agent.bsmpc import BSMPC
import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

import argparse
import numpy as np
import torch
import tqdm
import random
import wandb
import time


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def evaluate(env, agent, num_episodes, step, env_step, video=None):
    """Evaluate a trained agent and optionally save a video."""
    episode_rewards = []
    episode_successes = []
    for i in range(num_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        if video:
            video.init(env, enabled=(i == 0))
        while not done:
            action = agent.get_action(obs, eval_mode=True, step=step, t0=t == 0)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            if video:
                video.record(env)
            t += 1
        episode_rewards.append(ep_reward)
        episode_successes.append(info.get('success', 0))
        if video:
            video.save(env_step)
    return np.nanmean(episode_rewards), np.nanmean(episode_successes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--disable_wandb", dest="disable_wandb", action="store_true")
    parser.add_argument("--bisim_coef", type=float)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--add_noise", dest="add_noise", action="store_true")

    cfg = parser.parse_args()
    if cfg.config:
        cfg = parse_cfg(cfg, cfg.config)
    cfg.device = cfg.device if torch.cuda.is_available() else "cpu"

    # set seed
    set_seed(cfg.seed)

    env = dm_control.make_env(cfg)
    project_name = "IMAGE_BSMPC" if cfg.modality == 'pixels' else "BSMPC"

    # create replay buffer
    replay_buffer = ReplayBuffer(cfg)

    # create agent
    if cfg.agent_type == "TDMPC":
        agent = TDMPC(cfg)
    elif cfg.agent_type == "BSMPC":
        agent = BSMPC(cfg)
    else:
        raise NotImplementedError

    # setup wandb
    mode = "disabled" if cfg.disable_wandb else "online"
    group_name = cfg.agent_type + "_" + cfg.task
    if cfg.video_path is not None:
        group_name += "_distract"
    elif cfg.add_noise:
        group_name += "_noise"
    wandb.init(project=project_name, config=cfg, group=group_name, mode=mode)
    wandb.run.name = f"{group_name}_seed{cfg.seed}"
    if cfg.agent_type == "BSMPC":
        wandb.run.name += f"_bisim_coef{cfg.bisim_coef}"
    if not cfg.disable_wandb:
        wandb.mark_preempting()

    episode_idx = 0
    start_time = time.time()
    for step in tqdm.tqdm(
        range(0, cfg.train_steps + cfg.episode_length, cfg.episode_length)
    ):
        # 1, collect episodes
        obs = env.reset()
        episode = []
        done = False
        while not done:
            if step > cfg.seed_steps:
                action = agent.get_action(obs, step, t0=len(episode) == 1)
                action = action.detach().cpu()
            else:
                action = env.rand_act()

            next_obs, reward, done, _ = env.step(action)

            episode.append(convert_dict(env, obs, action, reward))
            obs = next_obs

        assert len(episode) == cfg.episode_length
        episode.append(convert_dict(env, obs))
        replay_buffer.add(torch.cat(episode))

        # 2. train the agent
        """
        obs: (horizon + 1, batch_size, shape)
        action: (horizon, batch_size, action_dim)
        reward: (horizon, batch_size, 1)
        """
        if step >= cfg.seed_steps:
            num_updates = (
                cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
            )
            for _ in range(num_updates):
                obs, action, reward = replay_buffer.sample()
                log_data = agent.update(obs, action, reward, step)
            wandb.log(log_data)

        episode_idx += 1
        env_step = int(step * cfg.action_repeat)

        if step % cfg.eval_freq == 0:
            episode_reward, episode_success = evaluate(
                env, agent, cfg.eval_episodes, step, env_step
            )
            print(f"environmental step: {env_step}  episode_reward: {episode_reward}")
            print(f"environmental step: {env_step}  episode_success: {episode_success}")
            wandb.log(
                {
                    "eval/env_step": env_step,
                    "eval/return": episode_reward,
                    "eval/success": episode_success,
                    "total_time:": time.time() - start_time,
                }
            )
    wandb.finish()


if __name__ == "__main__":
    main()
