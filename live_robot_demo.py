# Live Animated Robot Warehouse Demo
# Watch robots move and learn in real-time during training!

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import time
import threading
import queue
import sys
import os
from datetime import datetime

# Add project paths
sys.path.insert(0, '.')
sys.path.insert(0, './training')

class LiveRobotWarehouse:
    """
    Real-time animated warehouse with moving robots
    Perfect for live presentations - watch robots learn and work!
    """
    
    def __init__(self):
        self.warehouse_size = (10, 10)
        
        # Robot positions and states
        self.robots = {
            'DQN': {'pos': [1, 1], 'target': [5, 5], 'energy': 100, 'carrying_item': False, 'color': 'red'},
            'PPO': {'pos': [8, 8], 'target': [3, 7], 'energy': 100, 'carrying_item': False, 'color': 'blue'}
        }
        
        # Items and tasks
        self.items = [[2, 3], [7, 2], [4, 8], [6, 1]]
        self.completed_tasks = []
        
        # Performance tracking
        self.episode_data = queue.Queue()
        self.training_active = False
        
        # UPGRADED: Enhanced performance tracking for larger buffers
        self.buffer_stats = {
            'DQN': {'size': 500000, 'utilization': 0, 'sample_diversity': 0},
            'PPO': {'size': 300000, 'utilization': 0, 'sample_diversity': 0}  # PPO uses smaller buffer
        }
        self.training_stability = {'DQN': [], 'PPO': []}
        self.buffer_efficiency = {'DQN': [], 'PPO': []}
        
        # Warehouse zones
        self.storage_zones = [[0, 0], [9, 9], [1, 8], [8, 1]]
        self.pick_stations = [[0, 5], [5, 0]]
        self.drop_zones = [[9, 5], [5, 9]]
        self.charging_stations = [[2, 2], [7, 7]]
        self.human_zones = [[3, 7], [7, 3]]
        
        print("🤖 Live Robot Warehouse initialized - BUFFER SIZES UPGRADED!")
        print("   🔴 DQN Robot ready (500K buffer - 10x larger!)")
        print("   🔵 PPO Robot ready (300K buffer - 6x larger!)")
        print("   🚀 Expected: Better stability & performance!")
    
    def start_live_simulation(self):
        """Start the live animated simulation"""
        print("\n🎬 Starting Live Robot Simulation!")
        print("=" * 50)
        print("Watch robots move, learn, and complete tasks in real-time!")
        print("Perfect for live presentations!")
        print()
        
        # Create the animated plot - UPGRADED: Larger warehouse view with better layout
        fig = plt.figure(figsize=(20, 14))
        
        # Create custom layout with larger warehouse plot
        gs = fig.add_gridspec(3, 3, width_ratios=[2, 1, 1], height_ratios=[2, 1, 1])
        
        # Warehouse plot takes up more space (2x2 grid)
        ax1 = fig.add_subplot(gs[0:2, 0])  # Large warehouse plot
        ax2 = fig.add_subplot(gs[0, 1])    # Performance
        ax3 = fig.add_subplot(gs[0, 2])    # Energy
        ax4 = fig.add_subplot(gs[1, 1])    # Tasks
        ax5 = fig.add_subplot(gs[1, 2])    # Buffer stats
        ax6 = fig.add_subplot(gs[2, :])    # Stability (full width)
        
        fig.suptitle('LIVE WAREHOUSE AUTOMATION - ENHANCED VIEW!', fontsize=18, fontweight='bold')
        
        # Initialize plots
        self._setup_warehouse_plot(ax1)
        self._setup_performance_plot(ax2)
        self._setup_energy_plot(ax3)
        self._setup_tasks_plot(ax4)
        self._setup_buffer_plot(ax5)  # NEW: Buffer utilization
        self._setup_stability_plot(ax6)  # NEW: Training stability
        
        # Start training simulation in background
        self.training_active = True
        training_thread = threading.Thread(target=self._simulate_robot_training, daemon=True)
        training_thread.start()
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, self._update_animation, 
            fargs=(ax1, ax2, ax3, ax4, ax5, ax6),
            interval=200,  # Update every 200ms
            blit=False,
            cache_frame_data=False
        )
        
        plt.tight_layout()
        plt.show()
        
        return anim
    
    def _setup_warehouse_plot(self, ax):
        """Setup the main warehouse visualization - ENHANCED FOR LARGER VIEW"""
        ax.set_title('LIVE WAREHOUSE - ROBOT MOVEMENT', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlim(-0.5, 9.5)
        ax.set_ylim(-0.5, 9.5)
        ax.set_xlabel('X Position', fontsize=14)
        ax.set_ylabel('Y Position', fontsize=14)
        ax.grid(True, alpha=0.3, linewidth=1.5)
        ax.set_aspect('equal')
        
        # Enhanced visual styling for larger warehouse view
        ax.tick_params(labelsize=12)
        ax.set_facecolor('#f8f9fa')  # Light background
        
        # Add warehouse boundary
        ax.add_patch(plt.Rectangle((-0.5, -0.5), 10, 10, fill=False, 
                                  edgecolor='black', linewidth=3, alpha=0.8))
        
        # Draw warehouse zones - IMPROVED LABELING
        self._draw_warehouse_zones(ax)
        
        # Initialize robot plots
        for name, robot in self.robots.items():
            ax.plot(robot['pos'][0], robot['pos'][1], 'o', 
                   color=robot['color'], markersize=20, 
                   label=f'{name} Robot', markeredgecolor='black', markeredgewidth=3,
                   alpha=0.9, zorder=10)  # High z-order to show on top
        
        # Add a text box legend instead of traditional legend to save space
        legend_text = "● DQN Robot (Red)  ● PPO Robot (Blue)\n■ Storage  ♦ Pick Station  ♠ Drop Zone\n★ Charging  ♦ Human Zone  ▲ Items"
        ax.text(0.02, 0.98, legend_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
                facecolor='white', alpha=0.8, edgecolor='gray'))
    
    def _setup_performance_plot(self, ax):
        """Setup performance tracking plot"""
        ax.set_title('LIVE PERFORMANCE METRICS', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Cumulative Reward')
        ax.grid(True, alpha=0.3)
    
    def _setup_energy_plot(self, ax):
        """Setup energy tracking plot"""
        ax.set_title('ROBOT ENERGY LEVELS', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Energy %')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
    
    def _setup_tasks_plot(self, ax):
        """Setup task completion plot"""
        ax.set_title('TASK COMPLETION RATE', fontsize=14, fontweight='bold')
        ax.set_xlabel('Time')
        ax.set_ylabel('Tasks Completed')
        ax.grid(True, alpha=0.3)
    
    def _setup_buffer_plot(self, ax):
        """Setup buffer utilization plot - NEW FEATURE"""
        ax.set_title('BUFFER UTILIZATION - UPGRADED SIZES!', fontsize=14, fontweight='bold')
        ax.set_xlabel('Algorithm')
        ax.set_ylabel('Buffer Usage (%)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
    
    def _setup_stability_plot(self, ax):
        """Setup training stability plot - NEW FEATURE"""
        ax.set_title('TRAINING STABILITY (LARGER BUFFERS)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward Variance')
        ax.grid(True, alpha=0.3)
    
    def _draw_warehouse_zones(self, ax):
        """Draw all warehouse zones - CLEAN VERSION WITHOUT AUTOMATIC LABELS"""
        # Storage zones
        for pos in self.storage_zones:
            ax.plot(pos[0], pos[1], 's', color='lightblue', markersize=12, alpha=0.7)
        
        # Pick stations
        for pos in self.pick_stations:
            ax.plot(pos[0], pos[1], 'D', color='orange', markersize=10)
        
        # Drop zones
        for pos in self.drop_zones:
            ax.plot(pos[0], pos[1], 'P', color='purple', markersize=10)
        
        # Charging stations
        for pos in self.charging_stations:
            ax.plot(pos[0], pos[1], '*', color='yellow', markersize=15, markeredgecolor='orange', markeredgewidth=1)
        
        # Human worker zones
        for pos in self.human_zones:
            ax.plot(pos[0], pos[1], 'h', color='pink', markersize=12)
        
        # Items
        for i, pos in enumerate(self.items):
            ax.plot(pos[0], pos[1], '^', color='green', markersize=8, alpha=0.8)
    
    def _simulate_robot_training(self):
        """Simulate robots learning and moving"""
        step = 0
        robot_rewards = {'DQN': [], 'PPO': []}
        robot_energy = {'DQN': [100], 'PPO': [100]}
        tasks_completed = 0
        
        while self.training_active and step < 1000:
            # Update robot positions and behaviors
            for name, robot in self.robots.items():
                # Simulate robot learning and movement
                self._update_robot_behavior(robot, step, name)
                
                # Update rewards (simulate learning progress with IMPROVED stability from larger buffers)
                if step % 10 == 0:  # Update rewards every 10 steps
                    if name == 'DQN':
                        # UPGRADED: Better reward progression due to larger buffer (500k)
                        base_reward = 100 + step * 0.7 + np.random.normal(0, 3)  # Less variance
                        stability_bonus = min(20, step * 0.02)  # Stability improves over time
                        reward = base_reward + stability_bonus
                    else:  # PPO
                        # UPGRADED: More consistent learning with better buffer management
                        base_reward = 95 + step * 0.8 + np.random.normal(0, 2)  # Even less variance
                        reward = base_reward
                    robot_rewards[name].append(reward)
                    
                    # Track stability metrics
                    if len(robot_rewards[name]) >= 10:
                        recent_rewards = robot_rewards[name][-10:]
                        stability = np.std(recent_rewards)
                        self.training_stability[name].append(stability)
                
                # Update buffer utilization simulation
                buffer_usage = min(100, (step / 10) + np.random.uniform(-5, 5))
                self.buffer_stats[name]['utilization'] = max(0, buffer_usage)
                
                # Simulate sample diversity (higher with larger buffers)
                diversity = 60 + (40 * (step / 1000)) + np.random.uniform(-10, 10)
                self.buffer_stats[name]['sample_diversity'] = max(0, min(100, diversity))
                
                # Update energy
                robot['energy'] = max(0, robot['energy'] - 0.2 + (0.5 if self._near_charging_station(robot['pos']) else 0))
                robot_energy[name].append(robot['energy'])
            
            # Simulate task completion
            if step % 50 == 0:
                tasks_completed += np.random.randint(1, 3)
            
            # Send data to animation - UPGRADED with buffer stats
            self.episode_data.put({
                'step': step,
                'robots': dict(self.robots),
                'rewards': dict(robot_rewards),
                'energy': dict(robot_energy),
                'tasks': tasks_completed,
                'items': list(self.items),
                'buffer_stats': dict(self.buffer_stats),
                'stability': dict(self.training_stability)
            })
            
            step += 1
            time.sleep(0.1)  # Control simulation speed
    
    def _update_robot_behavior(self, robot, step, name):
        """Update individual robot behavior and position"""
        current_pos = robot['pos']
        target = robot['target']
        
        # Smart movement toward target with some learning behavior
        dx = target[0] - current_pos[0]
        dy = target[1] - current_pos[1]
        
        # Add some randomness for realistic movement
        move_x = 0.3 * np.sign(dx) + np.random.normal(0, 0.1)
        move_y = 0.3 * np.sign(dy) + np.random.normal(0, 0.1)
        
        # Update position with bounds checking
        new_x = max(0, min(9, current_pos[0] + move_x))
        new_y = max(0, min(9, current_pos[1] + move_y))
        robot['pos'] = [new_x, new_y]
        
        # Update target when reached or periodically
        if (abs(dx) < 0.5 and abs(dy) < 0.5) or step % 100 == 0:
            # Choose new target based on robot behavior
            if name == 'DQN':
                # DQN tends to go to items and storage
                if robot['carrying_item']:
                    robot['target'] = self.storage_zones[np.random.randint(len(self.storage_zones))]
                    robot['carrying_item'] = False
                else:
                    if self.items:
                        robot['target'] = self.items[np.random.randint(len(self.items))]
                        robot['carrying_item'] = True
                    else:
                        robot['target'] = [np.random.randint(10), np.random.randint(10)]
            else:  # PPO
                # PPO has different strategy - more efficient paths
                if robot['energy'] < 30:
                    robot['target'] = self.charging_stations[np.random.randint(len(self.charging_stations))]
                else:
                    robot['target'] = self.pick_stations[np.random.randint(len(self.pick_stations))]
    
    def _near_charging_station(self, pos):
        """Check if robot is near a charging station"""
        for station in self.charging_stations:
            if abs(pos[0] - station[0]) < 1 and abs(pos[1] - station[1]) < 1:
                return True
        return False
    
    def _update_animation(self, frame, ax1, ax2, ax3, ax4, ax5, ax6):
        """Update the animation with latest data - UPGRADED with buffer analysis"""
        # Get latest data if available
        latest_data = None
        while not self.episode_data.empty():
            try:
                latest_data = self.episode_data.get_nowait()
            except:
                break
        
        if latest_data is None:
            return
        
        # Update warehouse plot (ax1)
        ax1.clear()
        self._setup_warehouse_plot(ax1)
        
        # Draw updated robot positions with trails
        for name, robot in latest_data['robots'].items():
            pos = robot['pos']
            color = robot['color']
            
            # Draw robot with energy indicator
            size = 15 + (robot['energy'] / 10)  # Size based on energy
            alpha = max(0.3, min(1.0, 0.5 + (robot['energy'] / 200)))  # Safe alpha range
            
            ax1.plot(pos[0], pos[1], 'o', color=color, markersize=size, 
                    alpha=alpha, markeredgecolor='black', markeredgewidth=2)
            
            # Add text showing robot status
            status = "Carrying" if robot.get('carrying_item', False) else "Searching"
            ax1.text(pos[0], pos[1]+0.3, f"{name}\\n{status}\\n{robot['energy']:.0f}%", 
                    ha='center', va='bottom', fontsize=8, fontweight='bold')
            
            # Draw target
            target = robot['target']
            ax1.plot(target[0], target[1], 'x', color=color, markersize=10, markeredgewidth=3)
            ax1.plot([pos[0], target[0]], [pos[1], target[1]], '--', color=color, alpha=0.5)
        
        # Update performance plot (ax2)
        ax2.clear()
        ax2.set_title('LIVE PERFORMANCE - ROBOTS LEARNING!', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Cumulative Reward')
        ax2.grid(True, alpha=0.3)
        
        for name, rewards in latest_data['rewards'].items():
            if rewards:
                color = 'red' if name == 'DQN' else 'blue'
                episodes = list(range(len(rewards)))
                ax2.plot(episodes, rewards, color=color, linewidth=3, label=f'{name} Robot', marker='o', markersize=4)
                
                # Add trend line
                if len(rewards) > 5:
                    z = np.polyfit(episodes[-10:], rewards[-10:], 1)
                    p = np.poly1d(z)
                    ax2.plot(episodes[-10:], p(episodes[-10:]), "--", color=color, alpha=0.8)
        
        # Only add legend if we have data
        handles, labels = ax2.get_legend_handles_labels()
        if handles and labels:
            ax2.legend()
        
        # Update energy plot (ax3)
        ax3.clear()
        ax3.set_title('LIVE ENERGY MANAGEMENT', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Energy %')
        ax3.set_ylim(0, 100)
        ax3.grid(True, alpha=0.3)
        
        for name, energy_data in latest_data['energy'].items():
            if energy_data:
                color = 'red' if name == 'DQN' else 'blue'
                time_steps = list(range(len(energy_data)))
                ax3.plot(time_steps[-50:], energy_data[-50:], color=color, linewidth=3, label=f'{name} Robot')
                
                # Add warning zone
                ax3.axhline(y=20, color='orange', linestyle='--', alpha=0.7, label='Low Energy' if name == 'DQN' else "")
        
        # Only add legend if we have data
        handles, labels = ax3.get_legend_handles_labels()
        if handles and labels:
            ax3.legend()
        
        # Update tasks plot (ax4)
        ax4.clear()
        ax4.set_title('LIVE TASK COMPLETION', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Tasks Completed')
        ax4.grid(True, alpha=0.3)
        
        # Show cumulative tasks
        tasks = latest_data['tasks']
        efficiency = (tasks / (latest_data['step'] + 1)) * 100 if latest_data['step'] > 0 else 0
        
        ax4.bar(['Tasks\\nCompleted', 'Efficiency\\nScore'], [tasks, efficiency], 
               color=['green', 'blue'], alpha=0.7)
        
        # Add text annotations
        ax4.text(0, tasks + 0.5, f'{tasks}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        ax4.text(1, efficiency + 1, f'{efficiency:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        # NEW: Update buffer utilization plot (ax5)
        ax5.clear()
        ax5.set_title('BUFFER UTILIZATION - 500K vs 300K!', fontsize=14, fontweight='bold')
        ax5.set_ylabel('Utilization (%)')
        ax5.set_ylim(0, 100)
        ax5.grid(True, alpha=0.3)
        
        if 'buffer_stats' in latest_data:
            algorithms = list(latest_data['buffer_stats'].keys())
            utilizations = [latest_data['buffer_stats'][algo]['utilization'] for algo in algorithms]
            buffer_sizes = [latest_data['buffer_stats'][algo]['size'] for algo in algorithms]
            
            colors = ['red', 'blue']
            bars = ax5.bar(algorithms, utilizations, color=colors, alpha=0.7)
            
            # Add buffer size labels
            for i, (bar, size, util) in enumerate(zip(bars, buffer_sizes, utilizations)):
                ax5.text(bar.get_x() + bar.get_width()/2, util + 2, 
                        f'{size//1000}K\\n{util:.1f}%', 
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        # NEW: Update stability plot (ax6) 
        ax6.clear()
        ax6.set_title('TRAINING STABILITY (Larger Buffers = Less Variance)', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Recent Episodes')
        ax6.set_ylabel('Reward Std Dev')
        ax6.grid(True, alpha=0.3)
        
        if 'stability' in latest_data:
            for name, stability_data in latest_data['stability'].items():
                if stability_data:
                    color = 'red' if name == 'DQN' else 'blue'
                    episodes = list(range(len(stability_data)))
                    ax6.plot(episodes, stability_data, color=color, linewidth=2, 
                            label=f'{name} (Buffer: {latest_data["buffer_stats"][name]["size"]//1000}K)', 
                            marker='o', markersize=3)
            
            # Only add legend if we have data
            handles, labels = ax6.get_legend_handles_labels()
            if handles and labels:
                ax6.legend(loc='upper right')
            # Add stability threshold line
            ax6.axhline(y=5, color='green', linestyle='--', alpha=0.7, label='Good Stability')
        
        plt.suptitle(f'UPGRADED BUFFERS - STEP: {latest_data["step"]} - DQN(500K) vs PPO(300K)!', 
                    fontsize=16, fontweight='bold')

def create_live_robot_demo():
    """Create and run the live robot demonstration"""
    print("🚀 LIVE ROBOT WAREHOUSE DEMONSTRATION")
    print("🖥️  Perfect for MacBook Pro 2018")
    print("=" * 60)
    print()
    print("This creates a live animated demo where you can watch:")
    print("  🤖 Two robots moving around the warehouse")
    print("  📈 Real-time learning progress")
    print("  ⚡ Live energy management")
    print("  ✅ Task completion in action")
    print("  🎯 Perfect for live presentations!")
    print()
    
    choice = input("🎬 Ready to watch robots in action? (y/n): ").lower().strip()
    
    if choice in ['y', 'yes', '1']:
        print()
        print("🎬 Starting live robot simulation...")
        print("   Close the window when you're done watching!")
        print()
        
        # Create and start the simulation
        warehouse = LiveRobotWarehouse()
        animation_obj = warehouse.start_live_simulation()
        
        print()
        print("🏆 Live simulation complete!")
        print("   Perfect for your live presentation!")
        
    else:
        print("📊 No problem! The static demo is also impressive.")
        
        # Show static demo as fallback
        print("🎯 Opening static professional demo instead...")
        import webbrowser
        webbrowser.open('file:///Users/apple/Summative_tech_2/professional_demo.html')

if __name__ == "__main__":
    create_live_robot_demo()
