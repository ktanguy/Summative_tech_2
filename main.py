#!/usr/bin/env python3
"""
AI-Driven Warehouse Automation - Main Entry Point

This script demonstrates the best performing reinforcement learning model
for warehouse automation, showcasing how AI can enhance efficiency while
creating sustainable employment opportunities.

Mission: AI-Driven Automation in Warehouse Operations: Enhancing Efficiency 
and Creating Sustainable Employment Opportunities
"""

import os
import sys
import json
import gymnasium as gym
import numpy as np
import pygame
import time
from stable_baselines3 import DQN, PPO, A2C

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environment.custom_env import WarehouseEnvironment
from environment.rendering import Advanced3DRenderer

class WarehouseAutomationDemo:
    """
    Main demonstration class for AI-driven warehouse automation
    """
    
    def __init__(self):
        self.best_model = None
        self.best_algorithm = None
        self.env = None
        self.renderer_3d = None
        
        # Register custom environment
        gym.register(
            id="WarehouseEnv-v0",
            entry_point=lambda: WarehouseEnvironment(render_mode="human"),
            max_episode_steps=200
        )
    
    def load_best_model(self):
        """Load the best performing model from training results"""
        print("🔍 Searching for best trained model...")
        
        # Try to find saved results
        model_dirs = ['models/dqn/', 'models/pg/']
        results_files = ['results/', 'training/']
        
        best_score = -float('inf')
        best_model_path = None
        best_algo = None
        
        # Search for the best model based on available files
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                for file in os.listdir(model_dir):
                    if file.endswith('.zip'):
                        model_path = os.path.join(model_dir, file)
                        
                        # Try to determine algorithm from filename
                        if 'dqn' in file.lower():
                            try:
                                model = DQN.load(model_path)
                                best_model_path = model_path
                                best_algo = 'DQN'
                                break
                            except:
                                continue
                        elif 'ppo' in file.lower():
                            try:
                                model = PPO.load(model_path)
                                best_model_path = model_path
                                best_algo = 'PPO'
                                break
                            except:
                                continue
                        elif 'a2c' in file.lower():
                            try:
                                model = A2C.load(model_path)
                                best_model_path = model_path
                                best_algo = 'A2C'
                                break
                            except:
                                continue
        
        if best_model_path:
            print(f"✅ Found trained model: {best_algo}")
            print(f"📂 Model path: {best_model_path}")
            
            # Load the model
            if best_algo == 'DQN':
                self.best_model = DQN.load(best_model_path)
            elif best_algo == 'PPO':
                self.best_model = PPO.load(best_model_path)
            elif best_algo == 'A2C':
                self.best_model = A2C.load(best_model_path)
            
            self.best_algorithm = best_algo
            return True
        else:
            print("❌ No trained models found. Training a quick demo model...")
            return self._train_demo_model()
    
    def _train_demo_model(self):
        """Train a quick demonstration model if no saved models exist"""
        print("🚀 Training demonstration model (this may take a few minutes)...")
        
        # Create environment for training
        train_env = WarehouseEnvironment(render_mode=None)
        
        # Train a quick PPO model
        model = PPO(
            policy="MlpPolicy",
            env=train_env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            n_epochs=10,
            verbose=1
        )
        
        model.learn(total_timesteps=20000)
        
        # Save the demo model
        os.makedirs("models/demo/", exist_ok=True)
        model.save("models/demo/demo_ppo_model")
        
        self.best_model = model
        self.best_algorithm = "PPO (Demo)"
        
        train_env.close()
        print("✅ Demo model trained successfully!")
        return True
    
    def run_2d_demonstration(self, n_episodes=3):
        """Run 2D visualization demonstration"""
        print(f"\\n🎮 Starting 2D Demonstration ({n_episodes} episodes)")
        print("=" * 60)
        
        self.env = WarehouseEnvironment(render_mode="human")
        
        for episode in range(n_episodes):
            print(f"\\n📋 Episode {episode + 1}/{n_episodes}")
            obs, info = self.env.reset()
            total_reward = 0
            steps = 0
            
            print(f"🎯 Mission: Navigate warehouse, pick items, collaborate with humans")
            print(f"🤖 Algorithm: {self.best_algorithm}")
            print(f"⚡ Initial Energy: {info.get('robot_energy', 100)}")
            
            while steps < 200:
                # Predict action using trained model
                action, _ = self.best_model.predict(obs, deterministic=True)
                
                # Take action
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                steps += 1
                
                # Render environment
                self.env.render()
                
                # Add small delay for better visualization
                time.sleep(0.1)
                
                # Print key events
                if reward > 10:  # Significant positive reward
                    if reward > 40:
                        print(f"  ✅ Item picked/delivered! Reward: +{reward:.1f}")
                    elif reward > 4:
                        print(f"  🤝 Human collaboration! Reward: +{reward:.1f}")
                
                if terminated or truncated:
                    print(f"  🏁 Episode completed!")
                    break
            
            # Episode summary
            print(f"\\n📊 Episode {episode + 1} Results:")
            print(f"  Total Reward: {total_reward:.2f}")
            print(f"  Steps Taken: {steps}")
            print(f"  Items Delivered: {info.get('items_delivered', 0)}")
            print(f"  Energy Remaining: {info.get('robot_energy', 0)}")
            print(f"  Collaborative Actions: {info.get('collaborative_actions', 0)}")
            print(f"  Efficiency Score: {info.get('efficiency_score', 0):.2f}")
            
            # Wait for user input to continue
            if episode < n_episodes - 1:
                input("\\n⏸️  Press Enter to continue to next episode...")
        
        self.env.close()
    
    def run_3d_demonstration(self, n_episodes=2):
        """Run 3D visualization demonstration"""
        print(f"\\n🌟 Starting 3D Demonstration ({n_episodes} episodes)")
        print("=" * 60)
        print("🎮 Controls: Arrow keys to rotate camera, +/- to zoom")
        
        try:
            # Create environment without pygame rendering (we'll use OpenGL)
            self.env = WarehouseEnvironment(render_mode=None)
            
            # Initialize 3D renderer
            self.renderer_3d = Advanced3DRenderer(self.env)
            
            for episode in range(n_episodes):
                print(f"\\n🎬 3D Episode {episode + 1}/{n_episodes}")
                obs, info = self.env.reset()
                total_reward = 0
                steps = 0
                
                print(f"🎯 Watch the robot navigate the 3D warehouse environment")
                print(f"🤖 Algorithm: {self.best_algorithm}")
                
                running = True
                while running and steps < 200:
                    # Handle input events
                    running = self.renderer_3d.handle_input()
                    if not running:
                        break
                    
                    # Predict action
                    action, _ = self.best_model.predict(obs, deterministic=True)
                    
                    # Take action
                    obs, reward, terminated, truncated, info = self.env.step(action)
                    total_reward += reward
                    steps += 1
                    
                    # Update environment state in renderer
                    self.renderer_3d.env = self.env
                    
                    # Render 3D frame
                    self.renderer_3d.render_frame()
                    
                    # Small delay for smooth animation
                    time.sleep(0.05)
                    
                    if terminated or truncated:
                        print(f"  🎉 Episode completed in 3D!")
                        break
                
                print(f"\\n🎮 3D Episode {episode + 1} Results:")
                print(f"  Total Reward: {total_reward:.2f}")
                print(f"  Steps: {steps}")
                print(f"  Items Delivered: {info.get('items_delivered', 0)}")
                
                if episode < n_episodes - 1:
                    print("\\n⏸️  Starting next 3D episode in 3 seconds...")
                    time.sleep(3)
            
            self.renderer_3d.close()
            
        except Exception as e:
            print(f"❌ 3D rendering failed: {e}")
            print("💡 Falling back to 2D demonstration...")
            self.run_2d_demonstration(1)
    
    def run_random_agent_demo(self):
        """Run demonstration with random actions (no model)"""
        print("\\n🎲 Random Agent Demonstration (No Training)")
        print("=" * 60)
        print("This shows the environment before any AI training")
        
        env = WarehouseEnvironment(render_mode="human")
        obs, info = env.reset()
        
        print(f"🤖 Agent: Random Actions (No Intelligence)")
        print(f"📊 Purpose: Show warehouse environment components")
        
        total_reward = 0
        steps = 0
        
        while steps < 100:  # Shorter demo for random agent
            # Take random action
            action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            env.render()
            time.sleep(0.2)  # Slower for observation
            
            if terminated or truncated:
                break
        
        print(f"\\n📊 Random Agent Results:")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Steps Taken: {steps}")
        print(f"  Items Delivered: {info.get('items_delivered', 0)}")
        print(f"  Performance: Poor (as expected without learning)")
        
        env.close()
    
    def show_mission_context(self):
        """Display mission context and objectives"""
        print("\\n" + "=" * 80)
        print("🏭 AI-DRIVEN WAREHOUSE AUTOMATION DEMONSTRATION")
        print("=" * 80)
        print()
        print("📋 MISSION CONTEXT:")
        print("   AI-Driven Automation in Warehouse Operations: Enhancing Efficiency")
        print("   and Creating Sustainable Employment Opportunities")
        print()
        print("🎯 OBJECTIVES:")
        print("   • Optimize robot navigation and task performance")
        print("   • Demonstrate human-robot collaboration")
        print("   • Achieve energy-efficient operations")
        print("   • Show how AI creates new employment opportunities")
        print()
        print("🤖 ENVIRONMENT COMPONENTS:")
        print("   • Autonomous Robot: Learns optimal navigation")
        print("   • Storage Locations: Items to collect")
        print("   • Pick Stations: Task coordination points")
        print("   • Drop Zones: Delivery destinations")
        print("   • Human Workers: Collaborative partners")
        print("   • Charging Stations: Energy management")
        print()
        print("💼 EMPLOYMENT OPPORTUNITIES:")
        print("   • Robot maintenance and supervision")
        print("   • AI system monitoring and optimization")
        print("   • Human-robot collaboration coordination")
        print("   • Data analysis and performance improvement")
        print("   • Warehouse layout and workflow design")
        print()
        print("📈 SUCCESS METRICS:")
        print("   • Task completion rate")
        print("   • Energy efficiency")
        print("   • Human-robot collaboration score")
        print("   • Navigation efficiency")
        print("=" * 80)
    
    def interactive_menu(self):
        """Interactive menu for different demonstration modes"""
        while True:
            print("\\n🎮 DEMONSTRATION MENU")
            print("=" * 40)
            print("1. 📖 Show Mission Context")
            print("2. 🎲 Random Agent Demo (No AI)")
            print("3. 🎮 2D AI Agent Demo")
            print("4. 🌟 3D AI Agent Demo (Advanced)")
            print("5. 📊 Performance Analysis")
            print("6. ❌ Exit")
            print()
            
            choice = input("🎯 Select option (1-6): ").strip()
            
            try:
                if choice == '1':
                    self.show_mission_context()
                elif choice == '2':
                    self.run_random_agent_demo()
                elif choice == '3':
                    self.run_2d_demonstration()
                elif choice == '4':
                    self.run_3d_demonstration()
                elif choice == '5':
                    self.show_performance_analysis()
                elif choice == '6':
                    print("\\n👋 Thank you for exploring AI-driven warehouse automation!")
                    print("🚀 This technology can transform logistics while creating jobs!")
                    break
                else:
                    print("❌ Invalid choice. Please select 1-6.")
            except KeyboardInterrupt:
                print("\\n\\n⏸️  Demonstration interrupted by user.")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                print("💡 Continuing with menu...")
    
    def show_performance_analysis(self):
        """Show performance analysis of the AI agent"""
        print("\\n📊 AI AGENT PERFORMANCE ANALYSIS")
        print("=" * 50)
        
        if not self.best_model:
            print("❌ No trained model available for analysis.")
            return
        
        # Quick evaluation
        env = WarehouseEnvironment(render_mode=None)
        
        total_rewards = []
        success_rates = []
        energy_efficiency = []
        collaboration_scores = []
        
        print("🔍 Running performance evaluation...")
        
        for episode in range(10):
            obs, info = env.reset()
            total_reward = 0
            steps = 0
            
            while steps < 200:
                action, _ = self.best_model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                
                if terminated or truncated:
                    break
            
            total_rewards.append(total_reward)
            success_rates.append(1 if info.get('items_delivered', 0) > 0 else 0)
            energy_efficiency.append(info.get('robot_energy', 0) / 100)
            collaboration_scores.append(info.get('collaborative_actions', 0))
        
        env.close()
        
        # Display results
        print(f"\\n📈 RESULTS (10 episodes):")
        print(f"   Average Reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        print(f"   Success Rate: {np.mean(success_rates):.2f} ({sum(success_rates)}/10)")
        print(f"   Energy Efficiency: {np.mean(energy_efficiency):.2f}")
        print(f"   Collaboration Score: {np.mean(collaboration_scores):.2f}")
        print()
        print(f"💼 EMPLOYMENT IMPACT:")
        print(f"   • High collaboration score indicates need for human supervisors")
        print(f"   • Energy management requires maintenance technicians")
        print(f"   • Performance monitoring creates data analyst roles")
        print(f"   • System optimization needs AI engineers")
    
    def run(self):
        """Main execution function"""
        # Display welcome message
        print("🚀 AI-DRIVEN WAREHOUSE AUTOMATION SYSTEM")
        print("🎓 Summative Assignment - Mission Based Reinforcement Learning")
        print()
        
        # Load best model
        if not self.load_best_model():
            print("❌ Failed to load or train a model. Exiting...")
            return
        
        print(f"✅ AI Agent Ready: {self.best_algorithm}")
        print("🎯 Mission: Optimize warehouse operations while creating employment")
        
        # Start interactive demonstration
        self.interactive_menu()

def main():
    """Entry point for the warehouse automation demonstration"""
    try:
        demo = WarehouseAutomationDemo()
        demo.run()
    except KeyboardInterrupt:
        print("\\n\\n⏸️  Program interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\\n❌ Unexpected error: {e}")
        print("💡 Please check your environment setup and try again.")

if __name__ == "__main__":
    main()
