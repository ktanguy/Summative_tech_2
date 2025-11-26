import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from environment.custom_env import WarehouseEnvironment

class DQNTrainer:
    """
    Deep Q-Network trainer for warehouse automation
    Implements value-based reinforcement learning for robot navigation
    """
    
    def __init__(self, env_id="WarehouseEnv", model_save_path="models/dqn/"):
        self.env_id = env_id
        self.model_save_path = model_save_path
        os.makedirs(model_save_path, exist_ok=True)
        
        # Register custom environment
        gym.register(
            id=env_id,
            entry_point=lambda: WarehouseEnvironment(render_mode=None)
        )
        
        self.training_metrics = {
            'episodes': [],
            'rewards': [],
            'episode_lengths': [],
            'success_rates': [],
            'energy_efficiency': [],
            'collaboration_score': []
        }
    
    def create_environment(self, n_envs=1):
        """Create vectorized environment for training"""
        return make_vec_env(self.env_id, n_envs=n_envs)
    
    def train_dqn(self, hyperparams=None, total_timesteps=50000):
        """
        Train DQN agent with specified hyperparameters
        """
        if hyperparams is None:
            hyperparams = self.get_default_hyperparams()
        
        print(f"Training DQN with hyperparameters: {hyperparams}")
        
        # Create training environment
        env = self.create_environment(n_envs=4)
        eval_env = self.create_environment(n_envs=1)
        
        # Create DQN model
        model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=hyperparams['learning_rate'],
            buffer_size=hyperparams['buffer_size'],
            learning_starts=hyperparams['learning_starts'],
            batch_size=hyperparams['batch_size'],
            tau=hyperparams['tau'],
            gamma=hyperparams['gamma'],
            train_freq=hyperparams['train_freq'],
            gradient_steps=hyperparams['gradient_steps'],
            target_update_interval=hyperparams['target_update_interval'],
            exploration_fraction=hyperparams['exploration_fraction'],
            exploration_initial_eps=hyperparams['exploration_initial_eps'],
            exploration_final_eps=hyperparams['exploration_final_eps'],
            max_grad_norm=hyperparams['max_grad_norm'],
            tensorboard_log="./dqn_warehouse_tensorboard/",
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
        
        # Train the model
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            log_interval=100,
            progress_bar=True
        )
        
        # Save final model
        model_name = f"dqn_warehouse_{self._get_param_string(hyperparams)}"
        model.save(os.path.join(self.model_save_path, model_name))
        
        # Evaluate final performance
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=10, deterministic=True
        )
        
        # Collect detailed metrics
        metrics = self._collect_detailed_metrics(model, eval_env)
        
        env.close()
        eval_env.close()
        
        return {
            'model': model,
            'mean_reward': mean_reward,
            'std_reward': std_reward,
            'hyperparams': hyperparams,
            'metrics': metrics,
            'model_path': os.path.join(self.model_save_path, model_name)
        }
    
    def get_default_hyperparams(self):
        """Get default hyperparameters for DQN - UPGRADED BUFFER SIZES"""
        return {
            'learning_rate': 1e-4,
            'buffer_size': 500000,  # UPGRADED: 10x larger for better stability
            'learning_starts': 5000,  # More initial exploration before training
            'batch_size': 64,  # Larger batches for better gradients
            'tau': 1.0,
            'gamma': 0.99,
            'train_freq': 4,
            'gradient_steps': 1,
            'target_update_interval': 1000,
            'exploration_fraction': 0.3,
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': 0.1,
            'max_grad_norm': 10,
            'net_arch': [256, 256]
        }
    
    def hyperparameter_search(self, n_trials=10):
        """
        Perform hyperparameter search for DQN
        Tests different combinations to find optimal settings
        """
        hyperparameter_configs = [
            # Config 1: Conservative learning - UPGRADED BUFFERS
            {
                'learning_rate': 5e-5,
                'buffer_size': 200000,  # 6.7x larger
                'batch_size': 64,
                'gamma': 0.95,
                'exploration_fraction': 0.4,
                'net_arch': [128, 128]
            },
            # Config 2: Aggressive learning - UPGRADED BUFFERS
            {
                'learning_rate': 2e-4,
                'buffer_size': 1000000,  # 14x larger for maximum stability
                'batch_size': 128,
                'gamma': 0.99,
                'exploration_fraction': 0.2,
                'net_arch': [256, 256]
            },
            # Config 3: Deep network - UPGRADED BUFFERS
            {
                'learning_rate': 1e-4,
                'buffer_size': 750000,  # 15x larger
                'batch_size': 64,
                'gamma': 0.98,
                'exploration_fraction': 0.3,
                'net_arch': [512, 256, 128]
            },
            # Config 4: High exploration - UPGRADED BUFFERS
            {
                'learning_rate': 1e-4,
                'buffer_size': 400000,  # 10x larger
                'batch_size': 32,
                'gamma': 0.99,
                'exploration_fraction': 0.5,
                'exploration_initial_eps': 1.0,
                'exploration_final_eps': 0.05,
                'net_arch': [256, 128]
            },
            # Config 5: Fast convergence - UPGRADED BUFFERS
            {
                'learning_rate': 3e-4,
                'buffer_size': 300000,  # 15x larger
                'batch_size': 256,  # Much larger batches
                'gamma': 0.99,
                'train_freq': 1,
                'target_update_interval': 500,
                'net_arch': [256, 256]
            }
        ]
        
        results = []
        
        for i, config in enumerate(hyperparameter_configs[:n_trials]):
            print(f"\\n=== Trial {i+1}/{min(n_trials, len(hyperparameter_configs))} ===")
            
            # Merge with default hyperparams
            full_config = {**self.get_default_hyperparams(), **config}
            
            try:
                result = self.train_dqn(full_config, total_timesteps=30000)
                results.append(result)
                
                print(f"Trial {i+1} completed:")
                print(f"  Mean Reward: {result['mean_reward']:.2f} ± {result['std_reward']:.2f}")
                print(f"  Success Rate: {result['metrics']['success_rate']:.2f}")
                print(f"  Energy Efficiency: {result['metrics']['energy_efficiency']:.2f}")
                
            except Exception as e:
                print(f"Trial {i+1} failed: {e}")
                continue
        
        # Find best configuration
        if results:
            best_result = max(results, key=lambda x: x['mean_reward'])
            print(f"\\n=== Best Configuration ===")
            print(f"Mean Reward: {best_result['mean_reward']:.2f}")
            print(f"Hyperparams: {best_result['hyperparams']}")
            
            # Save best model info
            self._save_hyperparameter_results(results)
            
            return best_result
        else:
            print("No successful trials completed!")
            return None
    
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
                obs, reward, done, info = env.step(action)
                steps += 1
                
                # Track collaborations from vectorized environment
                if hasattr(info, '__len__') and len(info) > 0:
                    env_info = info[0] if isinstance(info, list) else info
                    if isinstance(env_info, dict) and 'collaborative_actions' in env_info:
                        collaborations = env_info['collaborative_actions']
            
            # Calculate metrics
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
    
    def _get_param_string(self, hyperparams):
        """Generate string representation of hyperparameters"""
        return f"lr{hyperparams['learning_rate']}_bs{hyperparams['batch_size']}_gamma{hyperparams['gamma']}"
    
    def _save_hyperparameter_results(self, results):
        """Save hyperparameter search results"""
        import json
        
        # Prepare results for JSON serialization
        json_results = []
        for result in results:
            json_result = {
                'mean_reward': float(result['mean_reward']),
                'std_reward': float(result['std_reward']),
                'hyperparams': result['hyperparams'],
                'metrics': {k: float(v) if isinstance(v, (int, float)) else v 
                           for k, v in result['metrics'].items() 
                           if isinstance(v, (int, float, str))}
            }
            json_results.append(json_result)
        
        with open(os.path.join(self.model_save_path, 'hyperparameter_results.json'), 'w') as f:
            json.dump(json_results, f, indent=2)
    
    def visualize_training_results(self, results):
        """Create visualizations of training results"""
        if not results:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('DQN Training Results Analysis', fontsize=16)
        
        # Extract data
        rewards = [r['mean_reward'] for r in results]
        success_rates = [r['metrics']['success_rate'] for r in results]
        energy_efficiency = [r['metrics']['energy_efficiency'] for r in results]
        collaboration_scores = [r['metrics']['collaboration_score'] for r in results]
        
        trial_numbers = list(range(1, len(results) + 1))
        
        # Plot 1: Mean Rewards
        axes[0, 0].bar(trial_numbers, rewards, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('Mean Reward per Configuration')
        axes[0, 0].set_xlabel('Trial Number')
        axes[0, 0].set_ylabel('Mean Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Success Rate
        axes[0, 1].bar(trial_numbers, success_rates, color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('Task Success Rate')
        axes[0, 1].set_xlabel('Trial Number')
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Energy Efficiency
        axes[1, 0].bar(trial_numbers, energy_efficiency, color='orange', alpha=0.7)
        axes[1, 0].set_title('Energy Efficiency')
        axes[1, 0].set_xlabel('Trial Number')
        axes[1, 0].set_ylabel('Efficiency Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Collaboration Score
        axes[1, 1].bar(trial_numbers, collaboration_scores, color='purple', alpha=0.7)
        axes[1, 1].set_title('Human-Robot Collaboration Score')
        axes[1, 1].set_xlabel('Trial Number')
        axes[1, 1].set_ylabel('Collaboration Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_save_path, 'dqn_training_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main training function"""
    print("=== AI-Driven Warehouse Automation: DQN Training ===")
    
    trainer = DQNTrainer()
    
    # Perform hyperparameter search
    print("Starting hyperparameter optimization...")
    best_result = trainer.hyperparameter_search(n_trials=10)
    
    if best_result:
        print(f"\\nBest DQN model achieved mean reward of {best_result['mean_reward']:.2f}")
        print("Training completed successfully!")
        
        # Load and test the best model
        print("\\nTesting best model...")
        env = gym.make("WarehouseEnv")
        model = DQN.load(best_result['model_path'])
        
        # Run test episodes
        for episode in range(3):
            obs, _ = env.reset()
            total_reward = 0
            steps = 0
            
            print(f"\\nTest Episode {episode + 1}:")
            while steps < 200:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Steps: {steps}")
            print(f"  Items Delivered: {info.get('items_delivered', 0)}")
            print(f"  Energy Remaining: {info.get('robot_energy', 0)}")
        
        env.close()
    else:
        print("Training failed!")

if __name__ == "__main__":
    main()
