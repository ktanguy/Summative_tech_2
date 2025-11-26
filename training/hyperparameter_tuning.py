import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import optuna
import gymnasium as gym

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import relative to project root
sys.path.insert(0, os.path.join(project_root, 'training'))
from dqn_training import DQNTrainer
from pg_training import PolicyGradientTrainer
from environment.custom_env import WarehouseEnvironment

class ComprehensiveHyperparameterTuning:
    """
    Comprehensive hyperparameter tuning for all RL algorithms
    Uses Optuna for intelligent hyperparameter search
    """
    
    def __init__(self, results_dir="results/"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
        self.all_results = {
            'DQN': [],
            'PPO': [],
            'A2C': [],
            'REINFORCE': []
        }
        
        # Register environment
        gym.register(
            id="WarehouseEnv-v0",
            entry_point=lambda: WarehouseEnvironment(render_mode=None),
            max_episode_steps=200
        )
    
    def objective_dqn(self, trial):
        """Optuna objective function for DQN hyperparameter optimization"""
        # Suggest hyperparameters
        hyperparams = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'buffer_size': trial.suggest_categorical('buffer_size', [200000, 500000, 750000, 1000000]),  # UPGRADED BUFFER SIZES
            'learning_starts': trial.suggest_int('learning_starts', 2000, 10000),  # More initial exploration
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),  # Larger batch options
            'tau': trial.suggest_float('tau', 0.005, 1.0),
            'gamma': trial.suggest_float('gamma', 0.9, 0.999),
            'train_freq': trial.suggest_categorical('train_freq', [1, 4, 8]),
            'gradient_steps': trial.suggest_int('gradient_steps', 1, 4),
            'target_update_interval': trial.suggest_categorical('target_update_interval', [500, 1000, 2000]),
            'exploration_fraction': trial.suggest_float('exploration_fraction', 0.1, 0.5),
            'exploration_initial_eps': 1.0,
            'exploration_final_eps': trial.suggest_float('exploration_final_eps', 0.01, 0.2),
            'max_grad_norm': trial.suggest_float('max_grad_norm', 1.0, 50.0),
            'net_arch': trial.suggest_categorical('net_arch', [[128, 128], [256, 256], [512, 256], [256, 128, 64]])
        }
        
        trainer = DQNTrainer()
        try:
            result = trainer.train_dqn(hyperparams, total_timesteps=25000)
            
            # Composite score considering multiple metrics
            reward_score = result['mean_reward']
            success_score = result['metrics']['success_rate'] * 100
            efficiency_score = result['metrics']['energy_efficiency'] * 50
            collaboration_score = result['metrics']['collaboration_score'] * 25
            
            composite_score = reward_score + success_score + efficiency_score + collaboration_score
            
            # Store result for later analysis
            result['composite_score'] = composite_score
            self.all_results['DQN'].append(result)
            
            return composite_score
        
        except Exception as e:
            print(f"DQN trial failed: {e}")
            return -1000  # Large penalty for failed trials
    
    def objective_ppo(self, trial):
        """Optuna objective function for PPO hyperparameter optimization"""
        hyperparams = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096]),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'n_epochs': trial.suggest_int('n_epochs', 3, 20),
            'gamma': trial.suggest_float('gamma', 0.9, 0.999),
            'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 1.0),
            'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
            'clip_range_vf': trial.suggest_float('clip_range_vf', 0.1, 1.0) if trial.suggest_categorical('use_clip_range_vf', [True, False]) else None,
            'normalize_advantage': trial.suggest_categorical('normalize_advantage', [True, False]),
            'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.1),
            'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0),
            'max_grad_norm': trial.suggest_float('max_grad_norm', 0.1, 2.0),
            'net_arch': trial.suggest_categorical('net_arch', [[128, 128], [256, 256], [512, 256], [256, 128, 64]])
        }
        
        trainer = PolicyGradientTrainer()
        try:
            result = trainer.train_ppo(hyperparams, total_timesteps=25000)
            
            reward_score = result['mean_reward']
            success_score = result['metrics']['success_rate'] * 100
            efficiency_score = result['metrics']['energy_efficiency'] * 50
            collaboration_score = result['metrics']['collaboration_score'] * 25
            
            composite_score = reward_score + success_score + efficiency_score + collaboration_score
            
            result['composite_score'] = composite_score
            self.all_results['PPO'].append(result)
            
            return composite_score
        
        except Exception as e:
            print(f"PPO trial failed: {e}")
            return -1000
    
    def objective_a2c(self, trial):
        """Optuna objective function for A2C hyperparameter optimization"""
        hyperparams = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'n_steps': trial.suggest_int('n_steps', 3, 20),
            'gamma': trial.suggest_float('gamma', 0.9, 0.999),
            'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 1.0),
            'ent_coef': trial.suggest_float('ent_coef', 0.0, 0.1),
            'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0),
            'max_grad_norm': trial.suggest_float('max_grad_norm', 0.1, 2.0),
            'rms_prop_eps': trial.suggest_float('rms_prop_eps', 1e-8, 1e-4, log=True),
            'use_rms_prop': trial.suggest_categorical('use_rms_prop', [True, False]),
            'normalize_advantage': trial.suggest_categorical('normalize_advantage', [True, False]),
            'net_arch': trial.suggest_categorical('net_arch', [[128, 128], [256, 256], [512, 256], [128, 64]])
        }
        
        trainer = PolicyGradientTrainer()
        try:
            result = trainer.train_a2c(hyperparams, total_timesteps=25000)
            
            reward_score = result['mean_reward']
            success_score = result['metrics']['success_rate'] * 100
            efficiency_score = result['metrics']['energy_efficiency'] * 50
            collaboration_score = result['metrics']['collaboration_score'] * 25
            
            composite_score = reward_score + success_score + efficiency_score + collaboration_score
            
            result['composite_score'] = composite_score
            self.all_results['A2C'].append(result)
            
            return composite_score
        
        except Exception as e:
            print(f"A2C trial failed: {e}")
            return -1000
    
    def run_comprehensive_tuning(self, n_trials_per_algorithm=15):
        """
        Run comprehensive hyperparameter tuning for all algorithms
        """
        print("=== Starting Comprehensive Hyperparameter Tuning ===")
        print(f"Running {n_trials_per_algorithm} trials per algorithm")
        print(f"Total trials: {n_trials_per_algorithm * 3}")  # DQN, PPO, A2C
        
        # Configure Optuna to minimize noise
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        # DQN Optimization
        print("\\n🤖 Optimizing DQN Hyperparameters...")
        study_dqn = optuna.create_study(direction='maximize', study_name='DQN_optimization')
        study_dqn.optimize(self.objective_dqn, n_trials=n_trials_per_algorithm)
        
        print(f"DQN Best Score: {study_dqn.best_value:.2f}")
        print(f"DQN Best Params: {study_dqn.best_params}")
        
        # PPO Optimization
        print("\\n🎯 Optimizing PPO Hyperparameters...")
        study_ppo = optuna.create_study(direction='maximize', study_name='PPO_optimization')
        study_ppo.optimize(self.objective_ppo, n_trials=n_trials_per_algorithm)
        
        print(f"PPO Best Score: {study_ppo.best_value:.2f}")
        print(f"PPO Best Params: {study_ppo.best_params}")
        
        # A2C Optimization
        print("\\n⚡ Optimizing A2C Hyperparameters...")
        study_a2c = optuna.create_study(direction='maximize', study_name='A2C_optimization')
        study_a2c.optimize(self.objective_a2c, n_trials=n_trials_per_algorithm)
        
        print(f"A2C Best Score: {study_a2c.best_value:.2f}")
        print(f"A2C Best Params: {study_a2c.best_params}")
        
        # Train final REINFORCE models (simplified approach)
        print("\\n🎲 Training REINFORCE Models...")
        self._train_reinforce_variants()
        
        # Analyze and save results
        self._analyze_all_results()
        self._save_all_results()
        self._create_comprehensive_visualizations()
        
        return {
            'dqn_study': study_dqn,
            'ppo_study': study_ppo,
            'a2c_study': study_a2c,
            'best_overall': self._find_best_overall_model()
        }
    
    def _train_reinforce_variants(self):
        """Train multiple REINFORCE variants with different hyperparameters"""
        trainer = PolicyGradientTrainer()
        
        reinforce_configs = [
            {
                'learning_rate': 5e-4,
                'n_steps': 200,
                'ent_coef': 0.01,
                'gamma': 0.99,
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
                'gamma': 0.98,
                'net_arch': [128, 128]
            },
            {
                'learning_rate': 8e-4,
                'n_steps': 175,
                'ent_coef': 0.015,
                'gamma': 0.99,
                'max_grad_norm': 0.5,
                'net_arch': [256, 256]
            },
            {
                'learning_rate': 1.5e-3,
                'n_steps': 120,
                'ent_coef': 0.008,
                'gamma': 0.97,
                'max_grad_norm': 1.5,
                'net_arch': [512, 128]
            }
        ]
        
        for i, config in enumerate(reinforce_configs):
            try:
                print(f"  REINFORCE variant {i+1}/5")
                full_config = {**trainer.get_reinforce_hyperparams(), **config}
                result = trainer.train_reinforce(full_config, total_timesteps=20000)
                
                # Calculate composite score
                reward_score = result['mean_reward']
                success_score = result['metrics']['success_rate'] * 100
                efficiency_score = result['metrics']['energy_efficiency'] * 50
                collaboration_score = result['metrics']['collaboration_score'] * 25
                
                composite_score = reward_score + success_score + efficiency_score + collaboration_score
                result['composite_score'] = composite_score
                
                self.all_results['REINFORCE'].append(result)
                print(f"    Score: {composite_score:.2f}")
                
            except Exception as e:
                print(f"  REINFORCE variant {i+1} failed: {e}")
    
    def _analyze_all_results(self):
        """Analyze results across all algorithms"""
        print("\\n=== COMPREHENSIVE ANALYSIS ===")
        
        algorithm_summary = {}
        
        for algorithm, results in self.all_results.items():
            if results:
                scores = [r['composite_score'] for r in results]
                rewards = [r['mean_reward'] for r in results]
                success_rates = [r['metrics']['success_rate'] for r in results]
                energy_effs = [r['metrics']['energy_efficiency'] for r in results]
                collab_scores = [r['metrics']['collaboration_score'] for r in results]
                
                algorithm_summary[algorithm] = {
                    'n_trials': len(results),
                    'avg_composite_score': np.mean(scores),
                    'best_composite_score': max(scores),
                    'avg_reward': np.mean(rewards),
                    'best_reward': max(rewards),
                    'avg_success_rate': np.mean(success_rates),
                    'avg_energy_efficiency': np.mean(energy_effs),
                    'avg_collaboration_score': np.mean(collab_scores),
                    'score_std': np.std(scores)
                }
                
                print(f"\\n{algorithm}:")
                print(f"  Trials completed: {len(results)}")
                print(f"  Average Score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
                print(f"  Best Score: {max(scores):.2f}")
                print(f"  Average Reward: {np.mean(rewards):.2f}")
                print(f"  Best Reward: {max(rewards):.2f}")
                print(f"  Success Rate: {np.mean(success_rates):.2f}")
                print(f"  Energy Efficiency: {np.mean(energy_effs):.2f}")
        
        return algorithm_summary
    
    def _find_best_overall_model(self):
        """Find the best performing model across all algorithms"""
        best_model = None
        best_score = -float('inf')
        
        for algorithm, results in self.all_results.items():
            for result in results:
                if result['composite_score'] > best_score:
                    best_score = result['composite_score']
                    best_model = result
                    best_model['algorithm_name'] = algorithm
        
        if best_model:
            print(f"\\n🏆 BEST OVERALL MODEL:")
            print(f"  Algorithm: {best_model['algorithm_name']}")
            print(f"  Composite Score: {best_model['composite_score']:.2f}")
            print(f"  Mean Reward: {best_model['mean_reward']:.2f}")
            print(f"  Success Rate: {best_model['metrics']['success_rate']:.2f}")
            print(f"  Energy Efficiency: {best_model['metrics']['energy_efficiency']:.2f}")
            print(f"  Collaboration Score: {best_model['metrics']['collaboration_score']:.2f}")
        
        return best_model
    
    def _save_all_results(self):
        """Save all results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        all_results_serializable = {}
        for algorithm, results in self.all_results.items():
            all_results_serializable[algorithm] = []
            for result in results:
                serializable_result = {
                    'algorithm': result.get('algorithm', algorithm),
                    'mean_reward': float(result['mean_reward']),
                    'std_reward': float(result['std_reward']),
                    'composite_score': float(result['composite_score']),
                    'hyperparams': result['hyperparams'],
                    'metrics': {k: float(v) if isinstance(v, (int, float)) else v 
                               for k, v in result['metrics'].items() 
                               if isinstance(v, (int, float, str))}
                }
                all_results_serializable[algorithm].append(serializable_result)
        
        with open(os.path.join(self.results_dir, f'comprehensive_results_{timestamp}.json'), 'w') as f:
            json.dump(all_results_serializable, f, indent=2)
        
        # Create summary CSV
        summary_data = []
        for algorithm, results in self.all_results.items():
            for i, result in enumerate(results):
                summary_data.append({
                    'Algorithm': algorithm,
                    'Trial': i + 1,
                    'Mean_Reward': result['mean_reward'],
                    'Composite_Score': result['composite_score'],
                    'Success_Rate': result['metrics']['success_rate'],
                    'Energy_Efficiency': result['metrics']['energy_efficiency'],
                    'Collaboration_Score': result['metrics']['collaboration_score'],
                    'Learning_Rate': result['hyperparams'].get('learning_rate', 'N/A')
                })
        
        df = pd.DataFrame(summary_data)
        df.to_csv(os.path.join(self.results_dir, f'results_summary_{timestamp}.csv'), index=False)
        
        print(f"\\n📁 Results saved to {self.results_dir}")
    
    def _create_comprehensive_visualizations(self):
        """Create comprehensive visualizations of all results"""
        if not any(self.all_results.values()):
            print("No results to visualize")
            return
        
        # Create multiple visualization figures
        self._plot_algorithm_comparison()
        self._plot_performance_metrics()
        self._plot_hyperparameter_analysis()
        self._plot_score_distributions()
    
    def _plot_algorithm_comparison(self):
        """Plot comprehensive algorithm comparison"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Warehouse Automation RL Algorithms - Performance Comparison', fontsize=16)
        
        algorithms = []
        mean_rewards = []
        success_rates = []
        energy_effs = []
        collab_scores = []
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, (algorithm, results) in enumerate(self.all_results.items()):
            if results:
                algorithms.append(algorithm)
                mean_rewards.append(np.mean([r['mean_reward'] for r in results]))
                success_rates.append(np.mean([r['metrics']['success_rate'] for r in results]))
                energy_effs.append(np.mean([r['metrics']['energy_efficiency'] for r in results]))
                collab_scores.append(np.mean([r['metrics']['collaboration_score'] for r in results]))
        
        # Plot 1: Mean Rewards
        bars1 = axes[0, 0].bar(algorithms, mean_rewards, color=colors[:len(algorithms)], alpha=0.8)
        axes[0, 0].set_title('Average Mean Reward by Algorithm', fontsize=14)
        axes[0, 0].set_ylabel('Mean Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, mean_rewards):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Success Rates
        bars2 = axes[0, 1].bar(algorithms, success_rates, color=colors[:len(algorithms)], alpha=0.8)
        axes[0, 1].set_title('Task Success Rate by Algorithm', fontsize=14)
        axes[0, 1].set_ylabel('Success Rate')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].grid(True, alpha=0.3)
        
        for bar, value in zip(bars2, success_rates):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                           f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Energy Efficiency
        bars3 = axes[1, 0].bar(algorithms, energy_effs, color=colors[:len(algorithms)], alpha=0.8)
        axes[1, 0].set_title('Energy Efficiency by Algorithm', fontsize=14)
        axes[1, 0].set_ylabel('Energy Efficiency')
        axes[1, 0].grid(True, alpha=0.3)
        
        for bar, value in zip(bars3, energy_effs):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Collaboration Scores
        bars4 = axes[1, 1].bar(algorithms, collab_scores, color=colors[:len(algorithms)], alpha=0.8)
        axes[1, 1].set_title('Human-Robot Collaboration Score', fontsize=14)
        axes[1, 1].set_ylabel('Collaboration Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        for bar, value in zip(bars4, collab_scores):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'algorithm_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_performance_metrics(self):
        """Plot detailed performance metrics"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Create scatter plot of reward vs success rate
        colors = ['red', 'green', 'blue', 'orange']
        markers = ['o', 's', '^', 'D']
        
        for i, (algorithm, results) in enumerate(self.all_results.items()):
            if results:
                rewards = [r['mean_reward'] for r in results]
                success_rates = [r['metrics']['success_rate'] for r in results]
                
                ax.scatter(success_rates, rewards, 
                          c=colors[i], marker=markers[i], 
                          s=100, alpha=0.7, label=algorithm)
        
        ax.set_xlabel('Task Success Rate', fontsize=12)
        ax.set_ylabel('Mean Reward', fontsize=12)
        ax.set_title('Performance Correlation: Reward vs Success Rate', fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'performance_correlation.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_hyperparameter_analysis(self):
        """Plot hyperparameter sensitivity analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Hyperparameter Sensitivity Analysis', fontsize=16)
        
        # Analyze learning rate impact across algorithms
        for algorithm, results in self.all_results.items():
            if results and len(results) > 1:
                learning_rates = []
                scores = []
                
                for result in results:
                    lr = result['hyperparams'].get('learning_rate')
                    if lr is not None:
                        learning_rates.append(lr)
                        scores.append(result['composite_score'])
                
                if learning_rates:
                    axes[0, 0].scatter(learning_rates, scores, alpha=0.6, label=algorithm)
        
        axes[0, 0].set_xlabel('Learning Rate (log scale)')
        axes[0, 0].set_ylabel('Composite Score')
        axes[0, 0].set_title('Learning Rate Impact')
        axes[0, 0].set_xscale('log')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Additional hyperparameter plots can be added here
        # For now, fill remaining subplots with summary info
        
        axes[0, 1].text(0.1, 0.5, 'Hyperparameter\\nAnalysis\\nSummary', 
                       transform=axes[0, 1].transAxes, fontsize=14,
                       verticalalignment='center')
        axes[0, 1].axis('off')
        
        axes[1, 0].text(0.1, 0.5, 'Additional\\nAnalysis\\nPlots', 
                       transform=axes[1, 0].transAxes, fontsize=14,
                       verticalalignment='center')
        axes[1, 0].axis('off')
        
        axes[1, 1].text(0.1, 0.5, 'Model\\nComparison\\nMetrics', 
                       transform=axes[1, 1].transAxes, fontsize=14,
                       verticalalignment='center')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'hyperparameter_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_score_distributions(self):
        """Plot score distributions for each algorithm"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        all_scores = []
        labels = []
        
        for algorithm, results in self.all_results.items():
            if results:
                scores = [r['composite_score'] for r in results]
                all_scores.append(scores)
                labels.append(f"{algorithm} (n={len(scores)})")
        
        if all_scores:
            ax.boxplot(all_scores, labels=labels)
            ax.set_title('Composite Score Distribution by Algorithm', fontsize=14)
            ax.set_ylabel('Composite Score')
            ax.grid(True, alpha=0.3)
            
            # Add mean markers
            for i, scores in enumerate(all_scores):
                mean_score = np.mean(scores)
                ax.scatter(i + 1, mean_score, marker='D', s=100, c='red', zorder=3)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'score_distributions.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main function to run comprehensive hyperparameter tuning"""
    print("🚀 AI-Driven Warehouse Automation - Comprehensive Hyperparameter Tuning")
    print("=" * 80)
    
    tuner = ComprehensiveHyperparameterTuning()
    
    # Run the comprehensive tuning
    results = tuner.run_comprehensive_tuning(n_trials_per_algorithm=10)
    
    print("\\n" + "=" * 80)
    print("🎉 COMPREHENSIVE HYPERPARAMETER TUNING COMPLETED!")
    print("=" * 80)
    
    if results['best_overall']:
        print(f"\\n🏆 WINNER: {results['best_overall']['algorithm_name']}")
        print(f"📊 Score: {results['best_overall']['composite_score']:.2f}")
        print(f"🎯 Mean Reward: {results['best_overall']['mean_reward']:.2f}")
        print(f"✅ Success Rate: {results['best_overall']['metrics']['success_rate']:.2f}")
        
        print("\\n📈 All results have been saved and visualized!")
        print("📁 Check the 'results/' directory for detailed analysis files.")
    else:
        print("❌ No successful models found. Check your environment setup.")

if __name__ == "__main__":
    main()
