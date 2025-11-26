import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from environment.custom_env import WarehouseEnvironment

class PolicyGradientTrainer:
    """
    Policy Gradient trainer for warehouse automation
    Implements PPO, A2C, and custom REINFORCE algorithms
    """
    
    def __init__(self, env_id="WarehouseEnv", model_save_path="models/pg/"):
        self.env_id = env_id
        self.model_save_path = model_save_path
        os.makedirs(model_save_path, exist_ok=True)
        
        # Register custom environment
        gym.register(
            id=env_id,
            entry_point=lambda: WarehouseEnvironment(render_mode=None),
            max_episode_steps=200
        )
        
        self.algorithms = ['PPO', 'A2C', 'REINFORCE']
        
    def create_environment(self, n_envs=1):
        """Create vectorized environment for training"""
        return make_vec_env(self.env_id, n_envs=n_envs)
    
    def train_ppo(self, hyperparams=None, total_timesteps=50000):
        """Train PPO agent"""
        if hyperparams is None:
            hyperparams = self.get_ppo_hyperparams()
        
        print(f"Training PPO with hyperparameters: {hyperparams}")
        
        env = self.create_environment(n_envs=4)
        eval_env = self.create_environment(n_envs=1)
        
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=hyperparams['learning_rate'],
            n_steps=hyperparams['n_steps'],
            batch_size=hyperparams['batch_size'],
            n_epochs=hyperparams['n_epochs'],
            gamma=hyperparams['gamma'],
            gae_lambda=hyperparams['gae_lambda'],
            clip_range=hyperparams['clip_range'],
            clip_range_vf=hyperparams.get('clip_range_vf'),
            normalize_advantage=hyperparams.get('normalize_advantage', True),
            ent_coef=hyperparams['ent_coef'],
            vf_coef=hyperparams['vf_coef'],
            max_grad_norm=hyperparams['max_grad_norm'],
            tensorboard_log="./ppo_warehouse_tensorboard/",
            policy_kwargs=dict(net_arch=hyperparams['net_arch']),
            verbose=1,
            device='auto'
        )
        
        # Setup callbacks
        callback_on_best = StopTrainingOnRewardThreshold(
            reward_threshold=200, verbose=1
        )
        eval_callback = EvalCallback(
            eval_env,
            callback_on_new_best=callback_on_best,
            n_eval_episodes=5,
            eval_freq=5000,
            log_path="./logs/",
            best_model_save_path=self.model_save_path,
            verbose=1
        )
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            log_interval=10,
            progress_bar=True
        )
        
        model_name = f"ppo_warehouse_{self._get_param_string(hyperparams)}"
        model.save(os.path.join(self.model_save_path, model_name))
        
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=10, deterministic=True
        )
        
        metrics = self._collect_detailed_metrics(model, eval_env)
        
        env.close()
        eval_env.close()
        
        return {
            'algorithm': 'PPO',
            'model': model,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'hyperparams': hyperparams,
            'metrics': metrics,
            'model_path': os.path.join(self.model_save_path, model_name)
        }
    
    def train_a2c(self, hyperparams=None, total_timesteps=50000):
        """Train A2C agent"""
        if hyperparams is None:
            hyperparams = self.get_a2c_hyperparams()
        
        print(f"Training A2C with hyperparameters: {hyperparams}")
        
        env = self.create_environment(n_envs=4)
        eval_env = self.create_environment(n_envs=1)
        
        model = A2C(
            policy="MlpPolicy",
            env=env,
            learning_rate=hyperparams['learning_rate'],
            n_steps=hyperparams['n_steps'],
            gamma=hyperparams['gamma'],
            gae_lambda=hyperparams['gae_lambda'],
            ent_coef=hyperparams['ent_coef'],
            vf_coef=hyperparams['vf_coef'],
            max_grad_norm=hyperparams['max_grad_norm'],
            rms_prop_eps=hyperparams.get('rms_prop_eps', 1e-5),
            use_rms_prop=hyperparams.get('use_rms_prop', True),
            normalize_advantage=hyperparams.get('normalize_advantage', False),
            tensorboard_log="./a2c_warehouse_tensorboard/",
            policy_kwargs=dict(net_arch=hyperparams['net_arch']),
            verbose=1,
            device='auto'
        )
        
        callback_on_best = StopTrainingOnRewardThreshold(
            reward_threshold=200, verbose=1
        )
        eval_callback = EvalCallback(
            eval_env,
            callback_on_new_best=callback_on_best,
            n_eval_episodes=5,
            eval_freq=5000,
            log_path="./logs/",
            best_model_save_path=self.model_save_path,
            verbose=1
        )
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            log_interval=10,
            progress_bar=True
        )
        
        model_name = f"a2c_warehouse_{self._get_param_string(hyperparams)}"
        model.save(os.path.join(self.model_save_path, model_name))
        
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=10, deterministic=True
        )
        
        metrics = self._collect_detailed_metrics(model, eval_env)
        
        env.close()
        eval_env.close()
        
        return {
            'algorithm': 'A2C',
            'model': model,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'hyperparams': hyperparams,
            'metrics': metrics,
            'model_path': os.path.join(self.model_save_path, model_name)
        }
    
    def train_reinforce(self, hyperparams=None, total_timesteps=50000):
        """
        Custom REINFORCE implementation
        """
        if hyperparams is None:
            hyperparams = self.get_reinforce_hyperparams()
        
        print(f"Training REINFORCE with hyperparameters: {hyperparams}")
        
        # Use PPO with specific settings to approximate REINFORCE
        env = self.create_environment(n_envs=1)
        eval_env = self.create_environment(n_envs=1)
        
        # REINFORCE-like settings using PPO
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=hyperparams['learning_rate'],
            n_steps=hyperparams['n_steps'],  # Collect full episodes
            batch_size=hyperparams['n_steps'],  # Use all collected steps
            n_epochs=1,  # Single policy update per collection
            gamma=hyperparams['gamma'],
            gae_lambda=1.0,  # Pure Monte Carlo returns
            clip_range=1.0,  # No clipping to approximate vanilla policy gradient
            clip_range_vf=None,
            normalize_advantage=False,  # Don't normalize for pure REINFORCE
            ent_coef=hyperparams['ent_coef'],
            vf_coef=0.0,  # No value function updates
            max_grad_norm=hyperparams['max_grad_norm'],
            tensorboard_log="./reinforce_warehouse_tensorboard/",
            policy_kwargs=dict(net_arch=hyperparams['net_arch']),
            verbose=1,
            device='auto'
        )
        
        callback_on_best = StopTrainingOnRewardThreshold(
            reward_threshold=200, verbose=1
        )
        eval_callback = EvalCallback(
            eval_env,
            callback_on_new_best=callback_on_best,
            n_eval_episodes=5,
            eval_freq=10000,
            log_path="./logs/",
            best_model_save_path=self.model_save_path,
            verbose=1
        )
        
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            log_interval=10,
            progress_bar=True
        )
        
        model_name = f"reinforce_warehouse_{self._get_param_string(hyperparams)}"
        model.save(os.path.join(self.model_save_path, model_name))
        
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=10, deterministic=True
        )
        
        metrics = self._collect_detailed_metrics(model, eval_env)
        
        env.close()
        eval_env.close()
        
        return {
            'algorithm': 'REINFORCE',
            'model': model,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'hyperparams': hyperparams,
            'metrics': metrics,
            'model_path': os.path.join(self.model_save_path, model_name)
        }
    
    def get_ppo_hyperparams(self):
        """Default PPO hyperparameters"""
        return {
            'learning_rate': 3e-4,
            'n_steps': 2048,
            'batch_size': 64,
            'n_epochs': 10,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_range': 0.2,
            'clip_range_vf': None,
            'normalize_advantage': True,
            'ent_coef': 0.01,
            'vf_coef': 0.5,
            'max_grad_norm': 0.5,
            'net_arch': [256, 256]
        }
    
    def get_a2c_hyperparams(self):
        """Default A2C hyperparameters"""
        return {
            'learning_rate': 7e-4,
            'n_steps': 5,
            'gamma': 0.99,
            'gae_lambda': 1.0,
            'ent_coef': 0.01,
            'vf_coef': 0.25,
            'max_grad_norm': 0.5,
            'rms_prop_eps': 1e-5,
            'use_rms_prop': True,
            'normalize_advantage': False,
            'net_arch': [256, 256]
        }
    
    def get_reinforce_hyperparams(self):
        """Default REINFORCE hyperparameters"""
        return {
            'learning_rate': 1e-3,
            'n_steps': 200,  # Full episode length
            'gamma': 0.99,
            'ent_coef': 0.01,
            'max_grad_norm': 1.0,
            'net_arch': [128, 128]
        }
    
    def hyperparameter_search_all_algorithms(self, n_trials_per_algo=3):
        """
        Comprehensive hyperparameter search for all policy gradient algorithms
        """
        all_results = []
        
        # PPO configurations
        ppo_configs = [
            {
                'learning_rate': 1e-4,
                'n_steps': 1024,
                'batch_size': 32,
                'n_epochs': 5,
                'clip_range': 0.1,
                'ent_coef': 0.005,
                'net_arch': [128, 128]
            },
            {
                'learning_rate': 5e-4,
                'n_steps': 2048,
                'batch_size': 128,
                'n_epochs': 15,
                'clip_range': 0.3,
                'ent_coef': 0.02,
                'net_arch': [256, 256]
            },
            {
                'learning_rate': 3e-4,
                'n_steps': 4096,
                'batch_size': 64,
                'n_epochs': 10,
                'clip_range': 0.2,
                'ent_coef': 0.01,
                'gae_lambda': 0.9,
                'net_arch': [512, 256]
            }
        ]
        
        # A2C configurations
        a2c_configs = [
            {
                'learning_rate': 5e-4,
                'n_steps': 5,
                'ent_coef': 0.01,
                'vf_coef': 0.5,
                'net_arch': [128, 128]
            },
            {
                'learning_rate': 1e-3,
                'n_steps': 10,
                'ent_coef': 0.02,
                'vf_coef': 0.25,
                'gamma': 0.95,
                'net_arch': [256, 128]
            },
            {
                'learning_rate': 7e-4,
                'n_steps': 8,
                'ent_coef': 0.005,
                'vf_coef': 0.75,
                'gae_lambda': 0.9,
                'net_arch': [256, 256]
            }
        ]
        
        # REINFORCE configurations
        reinforce_configs = [
            {
                'learning_rate': 5e-4,
                'n_steps': 200,
                'ent_coef': 0.01,
                'net_arch': [128, 64]
            },
            {
                'learning_rate': 1e-3,
                'n_steps': 150,
                'ent_coef': 0.02,
                'gamma': 0.95,
                'net_arch': [256, 128]
            },
            {
                'learning_rate': 2e-3,
                'n_steps': 100,
                'ent_coef': 0.005,
                'max_grad_norm': 0.5,
                'net_arch': [128, 128]
            }
        ]
        
        # Train PPO models
        print("\\n=== Training PPO Models ===")
        for i, config in enumerate(ppo_configs[:n_trials_per_algo]):
            print(f"\\nPPO Trial {i+1}/{n_trials_per_algo}")
            full_config = {**self.get_ppo_hyperparams(), **config}
            try:
                result = self.train_ppo(full_config, total_timesteps=40000)
                all_results.append(result)
                print(f"PPO Trial {i+1} - Mean Reward: {result['mean_reward']:.2f}")
            except Exception as e:
                print(f"PPO Trial {i+1} failed: {e}")
        
        # Train A2C models
        print("\\n=== Training A2C Models ===")
        for i, config in enumerate(a2c_configs[:n_trials_per_algo]):
            print(f"\\nA2C Trial {i+1}/{n_trials_per_algo}")
            full_config = {**self.get_a2c_hyperparams(), **config}
            try:
                result = self.train_a2c(full_config, total_timesteps=40000)
                all_results.append(result)
                print(f"A2C Trial {i+1} - Mean Reward: {result['mean_reward']:.2f}")
            except Exception as e:
                print(f"A2C Trial {i+1} failed: {e}")
        
        # Train REINFORCE models
        print("\\n=== Training REINFORCE Models ===")
        for i, config in enumerate(reinforce_configs[:n_trials_per_algo]):
            print(f"\\nREINFORCE Trial {i+1}/{n_trials_per_algo}")
            full_config = {**self.get_reinforce_hyperparams(), **config}
            try:
                result = self.train_reinforce(full_config, total_timesteps=30000)
                all_results.append(result)
                print(f"REINFORCE Trial {i+1} - Mean Reward: {result['mean_reward']:.2f}")
            except Exception as e:
                print(f"REINFORCE Trial {i+1} failed: {e}")
        
        # Analyze results
        if all_results:
            self._analyze_algorithm_comparison(all_results)
            self._save_pg_results(all_results)
            
            # Find best overall model
            best_result = max(all_results, key=lambda x: x['mean_reward'])
            print(f"\\n=== Best Overall Model ===")
            print(f"Algorithm: {best_result['algorithm']}")
            print(f"Mean Reward: {best_result['mean_reward']:.2f}")
            print(f"Success Rate: {best_result['metrics']['success_rate']:.2f}")
            
            return all_results, best_result
        else:
            print("No successful training runs completed!")
            return [], None
    
    def _collect_detailed_metrics(self, model, env, n_episodes=10):
        """Collect detailed performance metrics"""
        metrics = {
            'success_rate': 0,
            'avg_steps': 0,
            'energy_efficiency': 0,
            'collaboration_score': 0,
            'task_completion_time': [],
            'energy_usage': [],
            'navigation_efficiency': []
        }
        
        successful_episodes = 0
        total_steps = 0
        total_energy_used = 0
        total_collaborations = 0
        
        for episode in range(n_episodes):
            obs = env.reset()
            done = False
            steps = 0
            initial_energy = 100
            collaborations = 0
            
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
                
                if hasattr(info, '__len__') and len(info) > 0:
                    env_info = info[0] if isinstance(info, list) else info
                    if 'collaborative_actions' in env_info:
                        collaborations = env_info['collaborative_actions']
            
            if hasattr(info, '__len__') and len(info) > 0:
                env_info = info[0] if isinstance(info, list) else info
                if 'items_delivered' in env_info and env_info['items_delivered'] > 0:
                    successful_episodes += 1
                
                if 'robot_energy' in env_info:
                    energy_used = initial_energy - env_info['robot_energy']
                    total_energy_used += energy_used
                    metrics['energy_usage'].append(energy_used)
                
                total_collaborations += collaborations
            
            total_steps += steps
            metrics['task_completion_time'].append(steps)
        
        metrics['success_rate'] = successful_episodes / n_episodes
        metrics['avg_steps'] = total_steps / n_episodes
        metrics['energy_efficiency'] = 1 - (total_energy_used / (n_episodes * 100))
        metrics['collaboration_score'] = total_collaborations / n_episodes
        metrics['navigation_efficiency'] = successful_episodes / max(1, total_steps / n_episodes)
        
        return metrics
    
    def _analyze_algorithm_comparison(self, results):
        """Analyze and compare different algorithms"""
        algorithms = {}
        
        for result in results:
            algo = result['algorithm']
            if algo not in algorithms:
                algorithms[algo] = []
            algorithms[algo].append(result)
        
        print("\\n=== Algorithm Comparison ===")
        for algo, algo_results in algorithms.items():
            if algo_results:
                mean_rewards = [r['mean_reward'] for r in algo_results]
                success_rates = [r['metrics']['success_rate'] for r in algo_results]
                energy_effs = [r['metrics']['energy_efficiency'] for r in algo_results]
                
                print(f"\\n{algo}:")
                print(f"  Average Mean Reward: {np.mean(mean_rewards):.2f} ± {np.std(mean_rewards):.2f}")
                print(f"  Average Success Rate: {np.mean(success_rates):.2f} ± {np.std(success_rates):.2f}")
                print(f"  Average Energy Efficiency: {np.mean(energy_effs):.2f} ± {np.std(energy_effs):.2f}")
                print(f"  Best Performance: {max(mean_rewards):.2f}")
    
    def _get_param_string(self, hyperparams):
        """Generate string representation of hyperparameters"""
        lr = hyperparams.get('learning_rate', 0)
        return f"lr{lr}"
    
    def _save_pg_results(self, results):
        """Save policy gradient results"""
        import json
        
        json_results = []
        for result in results:
            json_result = {
                'algorithm': result['algorithm'],
                'mean_reward': float(result['mean_reward']),
                'std_reward': float(result['std_reward']),
                'hyperparams': result['hyperparams'],
                'metrics': {k: float(v) if isinstance(v, (int, float)) else v 
                           for k, v in result['metrics'].items() 
                           if isinstance(v, (int, float, str))}
            }
            json_results.append(json_result)
        
        with open(os.path.join(self.model_save_path, 'pg_algorithm_results.json'), 'w') as f:
            json.dump(json_results, f, indent=2)
    
    def visualize_algorithm_comparison(self, results):
        """Create comprehensive visualizations comparing algorithms"""
        if not results:
            return
        
        algorithms = {}
        for result in results:
            algo = result['algorithm']
            if algo not in algorithms:
                algorithms[algo] = []
            algorithms[algo].append(result)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Policy Gradient Algorithms Comparison', fontsize=16)
        
        algo_names = list(algorithms.keys())
        colors = ['skyblue', 'lightgreen', 'orange', 'purple'][:len(algo_names)]
        
        # Plot 1: Mean Rewards
        mean_rewards = []
        std_rewards = []
        for algo in algo_names:
            rewards = [r['mean_reward'] for r in algorithms[algo]]
            mean_rewards.append(np.mean(rewards))
            std_rewards.append(np.std(rewards))
        
        axes[0, 0].bar(algo_names, mean_rewards, yerr=std_rewards, 
                      capsize=5, color=colors, alpha=0.7)
        axes[0, 0].set_title('Mean Reward Comparison')
        axes[0, 0].set_ylabel('Mean Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Success Rates
        success_rates = []
        for algo in algo_names:
            rates = [r['metrics']['success_rate'] for r in algorithms[algo]]
            success_rates.append(np.mean(rates))
        
        axes[0, 1].bar(algo_names, success_rates, color=colors, alpha=0.7)
        axes[0, 1].set_title('Task Success Rate')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Energy Efficiency
        energy_effs = []
        for algo in algo_names:
            effs = [r['metrics']['energy_efficiency'] for r in algorithms[algo]]
            energy_effs.append(np.mean(effs))
        
        axes[0, 2].bar(algo_names, energy_effs, color=colors, alpha=0.7)
        axes[0, 2].set_title('Energy Efficiency')
        axes[0, 2].set_ylabel('Efficiency Score')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Collaboration Scores
        collab_scores = []
        for algo in algo_names:
            scores = [r['metrics']['collaboration_score'] for r in algorithms[algo]]
            collab_scores.append(np.mean(scores))
        
        axes[1, 0].bar(algo_names, collab_scores, color=colors, alpha=0.7)
        axes[1, 0].set_title('Human-Robot Collaboration')
        axes[1, 0].set_ylabel('Collaboration Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Average Steps
        avg_steps = []
        for algo in algo_names:
            steps = [r['metrics']['avg_steps'] for r in algorithms[algo]]
            avg_steps.append(np.mean(steps))
        
        axes[1, 1].bar(algo_names, avg_steps, color=colors, alpha=0.7)
        axes[1, 1].set_title('Average Episode Steps')
        axes[1, 1].set_ylabel('Steps')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Best Performance Distribution
        best_rewards = []
        for algo in algo_names:
            rewards = [r['mean_reward'] for r in algorithms[algo]]
            best_rewards.append(max(rewards) if rewards else 0)
        
        axes[1, 2].bar(algo_names, best_rewards, color=colors, alpha=0.7)
        axes[1, 2].set_title('Best Model Performance')
        axes[1, 2].set_ylabel('Best Mean Reward')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_path, 'pg_algorithms_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main training function for policy gradient algorithms"""
    print("=== AI-Driven Warehouse Automation: Policy Gradient Training ===")
    
    trainer = PolicyGradientTrainer()
    
    # Train all policy gradient algorithms
    print("Starting comprehensive policy gradient training...")
    all_results, best_result = trainer.hyperparameter_search_all_algorithms(n_trials_per_algo=3)
    
    if best_result:
        print(f"\\nBest model: {best_result['algorithm']}")
        print(f"Mean reward: {best_result['mean_reward']:.2f}")
        
        # Visualize results
        trainer.visualize_algorithm_comparison(all_results)
        
        print("\\nPolicy gradient training completed successfully!")
    else:
        print("Training failed!")

if __name__ == "__main__":
    main()
