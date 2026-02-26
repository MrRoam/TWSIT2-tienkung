import os
import sys
import json
import numpy as np
import torch
from tqdm import tqdm
from termcolor import cprint

# 添加路径
sys.path.insert(0, r"D:\Desktop\TWIST2\legged_gym")
sys.path.insert(0, r"D:\Desktop\TWIST2\rsl_rl")

from legged_gym.envs import task_registry
from legged_gym.gym_utils import get_args

def evaluate_model(model_path, task_name, num_episodes=100, num_envs=50):
    """
    评估模型性能
    
    Args:
        model_path: 模型路径
        task_name: 任务名称（如 'g1_mimic'）
        num_episodes: 评估的 episode 数量
        num_envs: 并行环境数量
    """
    print("=" * 80)
    print(f"评估模型: {model_path}")
    print(f"任务: {task_name}")
    print(f"Episodes: {num_episodes}, Envs: {num_envs}")
    print("=" * 80)
    
    # 获取配置
    env_cfg, train_cfg = task_registry.get_cfgs(name=task_name)
    
    # 修改评估配置
    env_cfg.env.num_envs = num_envs
    env_cfg.env.debug_viz = False  # 不显示可视化界面
    env_cfg.env.episode_length_s = 20  # 每个 episode 20 秒
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 10
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True
    
    # 禁用噪声和随机化（评估时）
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 0
    env_cfg.domain_rand.max_push_vel_xy = 0.0
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.action_delay = False
    
    # 运动课程
    if hasattr(env_cfg, "motion"):
        env_cfg.motion.motion_curriculum = False
    
    cprint(f"创建环境...", "yellow")
    args = get_args()
    args.task = task_name
    env, _ = task_registry.make_env(name=task_name, args=args, env_cfg=env_cfg)
    
    cprint(f"加载模型...", "yellow")
    log_pth = os.path.dirname(model_path)
    train_cfg.runner.resume = True
    ppo_runner, _ = task_registry.make_alg_runner(
        log_root=log_pth, 
        env=env, 
        name=task_name, 
        args=args, 
        train_cfg=train_cfg
    )
    
    # 获取策略
    if_normalize = env_cfg.env.normalize_obs
    if if_normalize:
        try:
            normalizer = ppo_runner.get_normalizer(device=env.device)
            cprint("归一化器加载成功", "green")
        except:
            cprint("警告: 归一化器加载失败", "yellow")
            normalizer = None
    
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # 统计变量
    episode_rewards = []
    episode_lengths = []
    all_rewards = []  # 每个 step 的 reward
    
    # 运行评估
    cprint("开始评估...", "green")
    
    total_steps = 0
    episodes_completed = 0
    obs = env.get_observations()
    
    with tqdm(total=num_episodes, desc="Episodes") as pbar:
        while episodes_completed < num_episodes:
            # 前向传播
            if if_normalize and normalizer is not None:
                normalized_obs = normalizer.normalize(obs.detach())
            else:
                normalized_obs = obs.detach()
            
            actions = policy(normalized_obs, hist_encoding=True)
            
            # 环境步进
            obs, _, rews, dones, infos = env.step(actions.detach())
            
            # 记录 reward
            all_rewards.extend(rews.cpu().numpy().tolist())
            
            # 检查完成的 episodes
            for i, done in enumerate(dones.cpu().numpy()):
                if done:
                    # 计算这个 episode 的总 reward 和长度
                    if 'episode' in infos:
                        ep_dict = infos['episode']
                        if 'episode_reward' in ep_dict:
                            episode_rewards.append(ep_dict['episode_reward'][i].item())
                        else:
                            # 如果没有 episode_reward，累加最近的 reward
                            episode_rewards.append(rews[i].item())
                        
                        if 'episode_length' in ep_dict:
                            episode_lengths.append(ep_dict['episode_length'][i].item())
                    
                    episodes_completed += 1
                    pbar.update(1)
                    
                    if episodes_completed >= num_episodes:
                        break
            
            total_steps += num_envs
    
    # 计算统计结果
    print("\n" + "=" * 80)
    print("评估结果统计")
    print("=" * 80)
    
    if episode_rewards:
        episode_rewards = np.array(episode_rewards)
        episode_lengths = np.array(episode_lengths)
        
        print(f"\nEpisode Reward:")
        print(f"  Mean: {episode_rewards.mean():.4f}")
        print(f"  Std:  {episode_rewards.std():.4f}")
        print(f"  Min:  {episode_rewards.min():.4f}")
        print(f"  Max:  {episode_rewards.max():.4f}")
        print(f"  Median: {np.median(episode_rewards):.4f}")
        
        print(f"\nEpisode Length (steps):")
        print(f"  Mean: {episode_lengths.mean():.2f}")
        print(f"  Std:  {episode_lengths.std():.2f}")
        print(f"  Min:  {episode_lengths.min():.2f}")
        print(f"  Max:  {episode_lengths.max():.2f}")
        
        print(f"\nStep Reward:")
        print(f"  Mean: {np.mean(all_rewards):.4f}")
        print(f"  Std:  {np.std(all_rewards):.4f}")
    else:
        cprint("警告: 没有收集到完整的 episode 数据", "yellow")
        print(f"Step Reward Mean: {np.mean(all_rewards):.4f}")
    
    # 保存结果
    results = {
        'model_path': model_path,
        'task_name': task_name,
        'num_episodes': episodes_completed,
        'num_envs': num_envs,
        'episode_rewards_mean': float(episode_rewards.mean()) if episode_rewards else None,
        'episode_rewards_std': float(episode_rewards.std()) if episode_rewards else None,
        'episode_rewards_min': float(episode_rewards.min()) if episode_rewards else None,
        'episode_rewards_max': float(episode_rewards.max()) if episode_rewards else None,
        'episode_lengths_mean': float(episode_lengths.mean()) if episode_lengths else None,
        'episode_rewards': episode_rewards.tolist() if episode_rewards else [],
        'episode_lengths': episode_lengths.tolist() if episode_lengths else [],
        'all_rewards_mean': float(np.mean(all_rewards)),
    }
    
    # 保存到文件
    output_dir = os.path.join(os.path.dirname(model_path), "eval_results")
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"eval_{os.path.basename(model_path).replace('.pt', '.json')}")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n结果已保存到: {output_path}")
    
    print("\n" + "=" * 80)
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='评估训练好的模型')
    parser.add_argument('--task', type=str, default='g1_mimic', help='任务名称')
    parser.add_argument('--checkpoint', type=str, 
                        default='D:/Desktop/TWIST2/legged_gym/legged_gym/logs/model_1800.pt',
                        help='模型 checkpoint 路径')
    parser.add_argument('--num_episodes', type=int, default=100, help='评估的 episode 数量')
    parser.add_argument('--num_envs', type=int, default=50, help='并行环境数量')
    
    args = parser.parse_args()
    
    try:
        results = evaluate_model(
            model_path=args.checkpoint,
            task_name=args.task,
            num_episodes=args.num_episodes,
            num_envs=args.num_envs
        )
        
        cprint("\n评估完成!", "green")
        
    except Exception as e:
        cprint(f"评估失败: {e}", "red")
        import traceback
        traceback.print_exc()
