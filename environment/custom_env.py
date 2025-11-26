import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from typing import Optional, Tuple, Dict, List
import random

class WarehouseEnvironment(gym.Env):
    """
    AI-Driven Warehouse Automation Environment
    
    This environment simulates a modern warehouse where an autonomous robot
    learns to navigate, pick items, and collaborate with human workers to
    optimize operations while creating sustainable employment opportunities.
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    def __init__(self, render_mode: Optional[str] = None, size: int = 10):
        self.size = size  # Size of the warehouse grid (10x10)
        self.window_size = 800  # PyGame window size
        self.render_mode = render_mode
        
        # Define the observation space (robot position, items, energy, human positions)
        # Flattened grid + robot position + energy + task info
        self.observation_space = spaces.Box(
            low=0, high=10, shape=(size * size + 7,), dtype=np.float32
        )
        
        # Define action space: 0=Up, 1=Down, 2=Left, 3=Right, 4=Pick, 5=Drop, 6=Charge
        self.action_space = spaces.Discrete(7)
        
        # Initialize environment components
        self._init_warehouse_layout()
        
        # Pygame rendering
        self.window = None
        self.clock = None
        
        # Performance metrics
        self.total_tasks_completed = 0
        self.human_robot_collaborations = 0
        self.energy_efficiency_score = 0
        
    def _init_warehouse_layout(self):
        """Initialize warehouse layout with storage, pick stations, drop zones, etc."""
        # Storage locations (items to pick)
        self.storage_locations = [
            (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3),
            (7, 7), (7, 8), (8, 7), (8, 8)
        ]
        
        # Pick stations (where robots collect items for orders)
        self.pick_stations = [(3, 5), (6, 4), (5, 7)]
        
        # Drop zones (delivery areas)
        self.drop_zones = [(0, 9), (9, 0), (9, 9)]
        
        # Charging stations (energy replenishment)
        self.charging_stations = [(0, 0), (4, 4)]
        
        # Human worker zones (collaborative areas)
        self.human_zones = [(2, 5), (5, 2), (7, 6), (6, 8)]
        
        # Obstacles (shelves, machinery)
        self.obstacles = [(3, 3), (6, 6), (1, 7), (8, 2)]
        
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Robot position (starts at charging station)
        self._robot_pos = np.array([0, 0], dtype=np.int32)
        
        # Robot energy level (0-100)
        self._robot_energy = 100
        
        # Current task (item to pick and deliver)
        self._current_task = self._generate_new_task()
        
        # Items in warehouse (available for picking)
        self._items = {loc: random.choice(['electronics', 'clothing', 'food', 'books']) 
                      for loc in self.storage_locations}
        
        # Human workers positions (dynamic)
        self._human_positions = random.sample(self.human_zones, 2)
        
        # Performance tracking
        self.steps_taken = 0
        self.items_picked = 0
        self.items_delivered = 0
        self.collaborative_actions = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def _generate_new_task(self) -> Dict:
        """Generate a new task (pick item from storage, deliver to drop zone)"""
        pick_location = random.choice(self.storage_locations)
        drop_location = random.choice(self.drop_zones)
        priority = random.choice(['high', 'medium', 'low'])
        
        return {
            'pick_location': pick_location,
            'drop_location': drop_location,
            'priority': priority,
            'has_item': False
        }
    
    def step(self, action: int):
        """Execute action and return new state"""
        self.steps_taken += 1
        reward = 0
        terminated = False
        
        # Movement actions (0-3)
        if action < 4:
            reward += self._move_robot(action)
        
        # Pick action (4)
        elif action == 4:
            reward += self._pick_item()
        
        # Drop action (5)
        elif action == 5:
            reward += self._drop_item()
        
        # Charge action (6)
        elif action == 6:
            reward += self._charge_robot()
        
        # Consume energy for any action
        self._robot_energy = max(0, self._robot_energy - 1)
        
        # Check for human-robot collaboration
        if tuple(self._robot_pos) in self._human_positions:
            reward += 5  # Collaboration bonus
            self.collaborative_actions += 1
        
        # Energy penalty if low
        if self._robot_energy < 20:
            reward -= 2
        
        # Step penalty to encourage efficiency
        reward -= 0.1
        
        # Check termination conditions
        if self._robot_energy <= 0:
            reward -= 50  # Heavy penalty for running out of energy
            terminated = True
        
        if self.steps_taken >= 200:  # Maximum episode length
            terminated = True
        
        # Generate new task if current one is completed
        if (self._current_task['has_item'] and 
            tuple(self._robot_pos) in self.drop_zones):
            reward += 100  # Task completion bonus
            self.items_delivered += 1
            self._current_task = self._generate_new_task()
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, reward, terminated, False, info
    
    def _move_robot(self, action: int) -> float:
        """Move robot in specified direction"""
        direction_map = {0: [-1, 0], 1: [1, 0], 2: [0, -1], 3: [0, 1]}
        direction = np.array(direction_map[action])
        
        new_pos = self._robot_pos + direction
        
        # Check bounds and obstacles
        if (0 <= new_pos[0] < self.size and 
            0 <= new_pos[1] < self.size and
            tuple(new_pos) not in self.obstacles):
            self._robot_pos = new_pos
            
            # Reward for moving towards current objective
            if self._current_task['has_item']:
                target = self._current_task['drop_location']
            else:
                target = self._current_task['pick_location']
            
            # Calculate distance to target
            old_distance = np.linalg.norm(self._robot_pos - direction - np.array(target))
            new_distance = np.linalg.norm(self._robot_pos - np.array(target))
            
            if new_distance < old_distance:
                return 1  # Moving closer to target
            else:
                return -0.5  # Moving away from target
        else:
            return -5  # Collision penalty
    
    def _pick_item(self) -> float:
        """Pick item from current location"""
        current_pos = tuple(self._robot_pos)
        
        if (current_pos in self.storage_locations and 
            current_pos in self._items and
            not self._current_task['has_item']):
            
            # Check if this is the correct item for current task
            if current_pos == self._current_task['pick_location']:
                del self._items[current_pos]
                self._current_task['has_item'] = True
                self.items_picked += 1
                return 50  # Successful pick
            else:
                return -10  # Wrong item
        else:
            return -5  # Invalid pick action
    
    def _drop_item(self) -> float:
        """Drop item at current location"""
        current_pos = tuple(self._robot_pos)
        
        if (self._current_task['has_item'] and 
            current_pos in self.drop_zones and
            current_pos == self._current_task['drop_location']):
            
            self._current_task['has_item'] = False
            return 100  # Successful delivery
        else:
            return -5  # Invalid drop action
    
    def _charge_robot(self) -> float:
        """Charge robot at charging station"""
        current_pos = tuple(self._robot_pos)
        
        if current_pos in self.charging_stations:
            self._robot_energy = min(100, self._robot_energy + 20)
            return 10  # Charging reward
        else:
            return -5  # Invalid charging location
    
    def _get_obs(self) -> np.ndarray:
        """Get current observation"""
        # Create grid representation
        grid = np.zeros((self.size, self.size), dtype=np.float32)
        
        # Mark different elements
        # Robot position
        grid[self._robot_pos[0], self._robot_pos[1]] = 1
        
        # Items
        for pos in self._items.keys():
            grid[pos[0], pos[1]] = 2
        
        # Pick stations
        for pos in self.pick_stations:
            grid[pos[0], pos[1]] = 3
        
        # Drop zones
        for pos in self.drop_zones:
            grid[pos[0], pos[1]] = 4
        
        # Charging stations
        for pos in self.charging_stations:
            grid[pos[0], pos[1]] = 5
        
        # Human workers
        for pos in self._human_positions:
            grid[pos[0], pos[1]] = 6
        
        # Obstacles
        for pos in self.obstacles:
            grid[pos[0], pos[1]] = 7
        
        # Flatten grid and add additional info
        flat_grid = grid.flatten()
        
        # Additional state information
        additional_info = np.array([
            self._robot_pos[0] / self.size,  # Normalized robot x
            self._robot_pos[1] / self.size,  # Normalized robot y
            self._robot_energy / 100,        # Normalized energy
            self._current_task['pick_location'][0] / self.size,  # Target x
            self._current_task['pick_location'][1] / self.size,  # Target y
            self._current_task['drop_location'][0] / self.size,  # Drop x
            self._current_task['drop_location'][1] / self.size,  # Drop y
        ], dtype=np.float32)
        
        return np.concatenate([flat_grid, additional_info])
    
    def _get_info(self) -> Dict:
        """Get additional info about current state"""
        return {
            "robot_position": self._robot_pos,
            "robot_energy": self._robot_energy,
            "steps_taken": self.steps_taken,
            "items_picked": self.items_picked,
            "items_delivered": self.items_delivered,
            "collaborative_actions": self.collaborative_actions,
            "current_task": self._current_task,
            "efficiency_score": (self.items_delivered * 100) / max(1, self.steps_taken)
        }
    
    def render(self):
        """Render the warehouse environment"""
        if self.render_mode is None:
            return
        
        return self._render_frame()
    
    def _render_frame(self):
        """Render a single frame using pygame"""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # White background
        
        pix_square_size = self.window_size / self.size
        
        # Draw grid
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                (100, 100, 100),
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=1,
            )
            pygame.draw.line(
                canvas,
                (100, 100, 100),
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=1,
            )
        
        # Draw warehouse elements
        self._draw_elements(canvas, pix_square_size)
        
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
    
    def _draw_elements(self, canvas, pix_square_size):
        """Draw all warehouse elements"""
        # Draw storage locations (green squares)
        for pos in self.storage_locations:
            if pos in self._items:
                color = (0, 255, 0)  # Green for items
            else:
                color = (200, 255, 200)  # Light green for empty storage
            pygame.draw.rect(
                canvas,
                color,
                pygame.Rect(
                    pix_square_size * pos[1],
                    pix_square_size * pos[0],
                    pix_square_size,
                    pix_square_size,
                ),
            )
        
        # Draw pick stations (blue squares)
        for pos in self.pick_stations:
            pygame.draw.rect(
                canvas,
                (0, 0, 255),
                pygame.Rect(
                    pix_square_size * pos[1],
                    pix_square_size * pos[0],
                    pix_square_size,
                    pix_square_size,
                ),
            )
        
        # Draw drop zones (orange squares)
        for pos in self.drop_zones:
            pygame.draw.rect(
                canvas,
                (255, 165, 0),
                pygame.Rect(
                    pix_square_size * pos[1],
                    pix_square_size * pos[0],
                    pix_square_size,
                    pix_square_size,
                ),
            )
        
        # Draw charging stations (yellow squares)
        for pos in self.charging_stations:
            pygame.draw.rect(
                canvas,
                (255, 255, 0),
                pygame.Rect(
                    pix_square_size * pos[1],
                    pix_square_size * pos[0],
                    pix_square_size,
                    pix_square_size,
                ),
            )
        
        # Draw human workers (purple circles)
        for pos in self._human_positions:
            pygame.draw.circle(
                canvas,
                (128, 0, 128),
                (
                    int(pix_square_size * (pos[1] + 0.5)),
                    int(pix_square_size * (pos[0] + 0.5)),
                ),
                int(pix_square_size / 3),
            )
        
        # Draw obstacles (black squares)
        for pos in self.obstacles:
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * pos[1],
                    pix_square_size * pos[0],
                    pix_square_size,
                    pix_square_size,
                ),
            )
        
        # Draw robot (red circle)
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (
                int(pix_square_size * (self._robot_pos[1] + 0.5)),
                int(pix_square_size * (self._robot_pos[0] + 0.5)),
            ),
            int(pix_square_size / 4),
        )
        
        # Draw energy bar
        energy_bar_width = 200
        energy_bar_height = 20
        energy_ratio = self._robot_energy / 100
        
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(10, 10, energy_bar_width, energy_bar_height),
        )
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(10, 10, energy_bar_width * energy_ratio, energy_bar_height),
        )
    
    def close(self):
        """Clean up rendering"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
