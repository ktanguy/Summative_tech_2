#!/usr/bin/env python3
"""
Static Random Agent Demonstration
Shows the warehouse environment with an agent taking random actions
(No training involved - just environment visualization)
"""

import sys
import os
sys.path.insert(0, './environment')

import numpy as np
import matplotlib.pyplot as plt
import time
from custom_env import WarehouseEnvironment

class RandomAgentDemo:
    """
    Demonstrates the warehouse environment with a random agent
    Required for assignment: "Create a static file that shows the agent 
    taking random actions (not using a model) in the custom environment"
    """
    
    def __init__(self):
        self.env = WarehouseEnvironment()
        self.trajectory = []
        
    def run_random_demo(self, num_steps=50):
        """Run random agent for demonstration purposes"""
        print("🎲 Random Agent Warehouse Demonstration")
        print("=" * 50)
        print("Showing agent taking random actions in warehouse environment")
        print("(No training or learning involved - pure visualization)")
        print()
        
        # Reset environment
        obs, _ = self.env.reset()
        self.trajectory = []
        
        total_reward = 0
        
        print("📍 Initial State:")
        print(f"   Observation shape: {obs.shape}")
        print(f"   Robot Energy: {int(obs[-3] * 100)}%")  # Energy is normalized
        print(f"   Environment initialized successfully")
        print()
        
        print("🎯 Starting Random Actions...")
        
        for step in range(num_steps):
            # Take random action
            action = self.env.action_space.sample()
            action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'PICK', 'DROP', 'CHARGE']
            
            # Execute action
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            
            # Record trajectory
            self.trajectory.append({
                'step': step,
                'action': action_names[action],
                'position': [int(obs[-4] * 10), int(obs[-3] * 10)],  # Extract normalized position
                'energy': int(obs[-3] * 100),  # Extract normalized energy
                'reward': reward,
                'carrying': obs[-1] > 0.5  # Check if carrying item
            })
            
            # Print step info
            print(f"Step {step+1:2d}: {action_names[action]:5s} → "
                  f"Energy: {int(obs[-3] * 100):3.0f}% | "
                  f"Reward: {reward:+6.1f} | "
                  f"Action executed successfully")
            
            if terminated or truncated:
                print(f"\n🏁 Episode terminated after {step+1} steps")
                break
                
        print(f"\n📊 Demo Summary:")
        print(f"   Total Steps: {len(self.trajectory)}")
        print(f"   Total Reward: {total_reward:.1f}")
        print(f"   Average Reward: {total_reward/len(self.trajectory):.2f}")
        
        # Create visualization
        self.create_static_visualization()
        
    def create_static_visualization(self):
        """Create static visualization of the random agent's trajectory"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Random Agent Warehouse Demonstration\\n(No Training - Pure Environment Visualization)', 
                    fontsize=16, fontweight='bold')
        
        # 1. Warehouse Layout with Trajectory
        ax1.set_title('Warehouse Layout & Agent Trajectory', fontsize=14, fontweight='bold')
        ax1.set_xlim(-0.5, 9.5)
        ax1.set_ylim(-0.5, 9.5)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Draw warehouse zones
        storage_zones = [[0, 0], [9, 9], [1, 8], [8, 1]]
        pick_stations = [[0, 5], [5, 0]]
        drop_zones = [[9, 5], [5, 9]]
        charging_stations = [[2, 2], [7, 7]]
        human_zones = [[3, 7], [7, 3]]
        
        # Plot zones
        for pos in storage_zones:
            ax1.plot(pos[0], pos[1], 's', color='lightblue', markersize=15, alpha=0.7)
        for pos in pick_stations:
            ax1.plot(pos[0], pos[1], 'D', color='orange', markersize=12)
        for pos in drop_zones:
            ax1.plot(pos[0], pos[1], 'P', color='purple', markersize=12)
        for pos in charging_stations:
            ax1.plot(pos[0], pos[1], '*', color='gold', markersize=20)
        for pos in human_zones:
            ax1.plot(pos[0], pos[1], 'h', color='pink', markersize=15)
        
        # Plot trajectory
        positions = [step['position'] for step in self.trajectory]
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        
        ax1.plot(x_coords, y_coords, 'r-', alpha=0.6, linewidth=2, label='Agent Path')
        ax1.plot(x_coords[0], y_coords[0], 'go', markersize=12, label='Start')
        ax1.plot(x_coords[-1], y_coords[-1], 'ro', markersize=12, label='End')
        
        ax1.legend()
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        
        # 2. Action Distribution
        ax2.set_title('Random Action Distribution', fontsize=14, fontweight='bold')
        actions = [step['action'] for step in self.trajectory]
        action_counts = {}
        for action in actions:
            action_counts[action] = action_counts.get(action, 0) + 1
        
        bars = ax2.bar(action_counts.keys(), action_counts.values(), 
                      color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3'])
        ax2.set_ylabel('Count')
        ax2.set_xlabel('Action Type')
        for bar, count in zip(bars, action_counts.values()):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(count), ha='center', fontweight='bold')
        
        # 3. Energy Level Over Time
        ax3.set_title('Energy Management (Random Agent)', fontsize=14, fontweight='bold')
        steps = [step['step'] for step in self.trajectory]
        energy_levels = [step['energy'] for step in self.trajectory]
        ax3.plot(steps, energy_levels, 'b-', linewidth=3, marker='o', markersize=4)
        ax3.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Low Energy Warning')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Energy Level (%)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # 4. Reward Progression
        ax4.set_title('Reward Progression (Random Actions)', fontsize=14, fontweight='bold')
        rewards = [step['reward'] for step in self.trajectory]
        cumulative_rewards = np.cumsum(rewards)
        
        ax4.bar(steps, rewards, alpha=0.6, color='lightcoral', label='Step Rewards')
        ax4_twin = ax4.twinx()
        ax4_twin.plot(steps, cumulative_rewards, 'g-', linewidth=3, label='Cumulative Reward')
        
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Step Reward', color='red')
        ax4_twin.set_ylabel('Cumulative Reward', color='green')
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        
        plt.tight_layout()
        plt.savefig('/Users/apple/Summative_tech_2/random_agent_demo.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Static visualization saved as 'random_agent_demo.png'")

def main():
    """Run the random agent demonstration"""
    print("🎮 WAREHOUSE RANDOM AGENT DEMO")
    print("Required for Assignment: Static Environment Visualization")
    print()
    
    demo = RandomAgentDemo()
    demo.run_random_demo(num_steps=30)
    
    print("\n🎯 This demonstrates:")
    print("  ✅ Custom environment functionality")
    print("  ✅ Action space and observation space")
    print("  ✅ Reward structure in action")
    print("  ✅ Energy management system")
    print("  ✅ Warehouse zone interactions")
    print("  ✅ Static visualization (as required)")

if __name__ == "__main__":
    main()
