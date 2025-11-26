import pygame
import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import math

class Advanced3DRenderer:
    """
    Advanced 3D visualization for the warehouse environment using OpenGL
    Provides immersive visualization of robot navigation and warehouse operations
    """
    
    def __init__(self, warehouse_env, window_size=(1200, 800)):
        self.env = warehouse_env
        self.window_size = window_size
        self.camera_angle_x = 30
        self.camera_angle_y = 45
        self.camera_distance = 20
        self.camera_target = [5, 5, 0]
        
        # Initialize pygame and OpenGL
        pygame.init()
        self.screen = pygame.display.set_mode(window_size, pygame.DOUBLEBUF | pygame.OPENGL)
        pygame.display.set_caption("AI-Driven Warehouse Automation - 3D View")
        
        # Configure OpenGL
        self._setup_opengl()
        
        # Robot animation state
        self.robot_animation_phase = 0
        self.robot_height = 0.5
        
    def _setup_opengl(self):
        """Configure OpenGL settings for 3D rendering"""
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        
        # Set up lighting
        glLightfv(GL_LIGHT0, GL_POSITION, [10, 10, 10, 1])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1])
        
        # Set perspective
        glMatrixMode(GL_PROJECTION)
        gluPerspective(45, self.window_size[0] / self.window_size[1], 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        
        # Set background color
        glClearColor(0.1, 0.1, 0.2, 1.0)
    
    def render_frame(self):
        """Render a complete 3D frame of the warehouse"""
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Set camera position
        self._update_camera()
        
        # Draw warehouse floor
        self._draw_floor()
        
        # Draw warehouse elements
        self._draw_storage_locations()
        self._draw_pick_stations()
        self._draw_drop_zones()
        self._draw_charging_stations()
        self._draw_obstacles()
        self._draw_human_workers()
        
        # Draw robot with animation
        self._draw_robot_3d()
        
        # Draw UI elements
        self._draw_hud()
        
        pygame.display.flip()
        
    def _update_camera(self):
        """Update camera position for optimal viewing"""
        # Calculate camera position based on angles
        x = self.camera_target[0] + self.camera_distance * math.cos(math.radians(self.camera_angle_y)) * math.cos(math.radians(self.camera_angle_x))
        y = self.camera_target[1] + self.camera_distance * math.sin(math.radians(self.camera_angle_y)) * math.cos(math.radians(self.camera_angle_x))
        z = self.camera_target[2] + self.camera_distance * math.sin(math.radians(self.camera_angle_x))
        
        gluLookAt(x, y, z,  # Camera position
                  self.camera_target[0], self.camera_target[1], self.camera_target[2],  # Target
                  0, 0, 1)  # Up vector
    
    def _draw_floor(self):
        """Draw the warehouse floor with grid pattern"""
        glColor3f(0.8, 0.8, 0.9)
        glBegin(GL_QUADS)
        glVertex3f(0, 0, 0)
        glVertex3f(self.env.size, 0, 0)
        glVertex3f(self.env.size, self.env.size, 0)
        glVertex3f(0, self.env.size, 0)
        glEnd()
        
        # Draw grid lines
        glColor3f(0.6, 0.6, 0.7)
        glLineWidth(1)
        glBegin(GL_LINES)
        for i in range(self.env.size + 1):
            # Vertical lines
            glVertex3f(i, 0, 0.01)
            glVertex3f(i, self.env.size, 0.01)
            # Horizontal lines
            glVertex3f(0, i, 0.01)
            glVertex3f(self.env.size, i, 0.01)
        glEnd()
    
    def _draw_storage_locations(self):
        """Draw storage locations as 3D boxes with items"""
        for pos in self.env.storage_locations:
            x, y = pos
            if pos in self.env._items:
                # Draw storage shelf with item
                glColor3f(0.2, 0.8, 0.2)  # Green
                self._draw_cube(x + 0.5, y + 0.5, 0.75, 0.8, 0.8, 1.5)
                
                # Draw item on shelf
                item_type = self.env._items[pos]
                self._draw_item(x + 0.5, y + 0.5, 1.6, item_type)
            else:
                # Empty storage
                glColor3f(0.8, 1.0, 0.8)  # Light green
                self._draw_cube(x + 0.5, y + 0.5, 0.75, 0.8, 0.8, 1.5)
    
    def _draw_pick_stations(self):
        """Draw pick stations as blue platforms"""
        glColor3f(0.2, 0.2, 0.8)  # Blue
        for pos in self.env.pick_stations:
            x, y = pos
            self._draw_cube(x + 0.5, y + 0.5, 0.1, 0.9, 0.9, 0.2)
            
            # Add conveyor belt effect
            glColor3f(0.3, 0.3, 0.9)
            for i in range(3):
                offset = (i - 1) * 0.2
                self._draw_cube(x + 0.5, y + 0.5 + offset, 0.15, 0.8, 0.1, 0.1)
    
    def _draw_drop_zones(self):
        """Draw drop zones as orange platforms"""
        glColor3f(1.0, 0.6, 0.0)  # Orange
        for pos in self.env.drop_zones:
            x, y = pos
            self._draw_cube(x + 0.5, y + 0.5, 0.1, 0.9, 0.9, 0.2)
            
            # Add delivery chute
            glColor3f(0.8, 0.4, 0.0)
            self._draw_cylinder(x + 0.5, y + 0.5, 0.3, 0.2, 0.4)
    
    def _draw_charging_stations(self):
        """Draw charging stations as yellow platforms with energy indicators"""
        for pos in self.env.charging_stations:
            x, y = pos
            
            # Base platform
            glColor3f(1.0, 1.0, 0.2)  # Yellow
            self._draw_cube(x + 0.5, y + 0.5, 0.1, 0.9, 0.9, 0.2)
            
            # Charging post
            glColor3f(0.8, 0.8, 0.0)
            self._draw_cylinder(x + 0.5, y + 0.5, 0.5, 0.1, 0.8)
            
            # Energy indicator (animated)
            energy_phase = (pygame.time.get_ticks() / 1000) % 2
            if energy_phase < 1:
                glColor3f(0.0, 1.0, 0.0)  # Green
                self._draw_sphere(x + 0.5, y + 0.5, 1.0, 0.1)
    
    def _draw_obstacles(self):
        """Draw obstacles as dark grey structures"""
        glColor3f(0.2, 0.2, 0.2)  # Dark grey
        for pos in self.env.obstacles:
            x, y = pos
            # Large storage units or machinery
            self._draw_cube(x + 0.5, y + 0.5, 1.0, 0.9, 0.9, 2.0)
            
            # Add some detail
            glColor3f(0.1, 0.1, 0.1)
            self._draw_cube(x + 0.3, y + 0.3, 1.1, 0.2, 0.2, 0.2)
            self._draw_cube(x + 0.7, y + 0.7, 1.1, 0.2, 0.2, 0.2)
    
    def _draw_human_workers(self):
        """Draw human workers as animated figures"""
        for i, pos in enumerate(self.env._human_positions):
            x, y = pos
            
            # Animation phase for each worker
            worker_phase = (pygame.time.get_ticks() / 1500 + i) % 2
            bob_height = 0.1 * math.sin(worker_phase * math.pi)
            
            # Body (purple)
            glColor3f(0.6, 0.2, 0.6)
            self._draw_cylinder(x + 0.5, y + 0.5, 0.9 + bob_height, 0.15, 0.6)
            
            # Head
            glColor3f(0.8, 0.6, 0.4)  # Skin tone
            self._draw_sphere(x + 0.5, y + 0.5, 1.6 + bob_height, 0.12)
            
            # Safety vest
            glColor3f(1.0, 0.8, 0.0)  # Yellow
            self._draw_cube(x + 0.5, y + 0.5, 1.1 + bob_height, 0.3, 0.25, 0.4)
            
            # Work tools (tablet or scanner)
            glColor3f(0.1, 0.1, 0.1)
            self._draw_cube(x + 0.3, y + 0.5, 1.2 + bob_height, 0.1, 0.15, 0.02)
    
    def _draw_robot_3d(self):
        """Draw the robot with detailed 3D model and animations"""
        x, y = self.env._robot_pos
        
        # Robot animation (hovering effect)
        self.robot_animation_phase += 0.1
        hover_offset = 0.05 * math.sin(self.robot_animation_phase)
        
        robot_x = x + 0.5
        robot_y = y + 0.5
        robot_z = self.robot_height + hover_offset
        
        # Main body
        glColor3f(0.8, 0.1, 0.1)  # Red
        self._draw_cylinder(robot_x, robot_y, robot_z, 0.2, 0.3)
        
        # Top sensor dome
        glColor3f(0.9, 0.9, 0.9)  # Light grey
        self._draw_sphere(robot_x, robot_y, robot_z + 0.35, 0.15)
        
        # Wheels/base
        glColor3f(0.2, 0.2, 0.2)  # Dark grey
        for wheel_angle in [0, 120, 240]:
            wheel_x = robot_x + 0.15 * math.cos(math.radians(wheel_angle))
            wheel_y = robot_y + 0.15 * math.sin(math.radians(wheel_angle))
            self._draw_cylinder(wheel_x, wheel_y, 0.1, 0.05, 0.1)
        
        # Robotic arm (if carrying item)
        if self.env._current_task['has_item']:
            glColor3f(0.5, 0.5, 0.5)  # Grey
            # Arm base
            self._draw_cylinder(robot_x + 0.2, robot_y, robot_z + 0.1, 0.03, 0.2)
            # Arm extension
            self._draw_cylinder(robot_x + 0.3, robot_y, robot_z + 0.2, 0.02, 0.15)
            
            # Carried item
            item_color = self._get_item_color(self.env._items.get(self.env._current_task['pick_location'], 'unknown'))
            glColor3fv(item_color)
            self._draw_cube(robot_x + 0.35, robot_y, robot_z + 0.3, 0.1, 0.1, 0.1)
        
        # LED status lights based on energy
        if self.env._robot_energy > 60:
            glColor3f(0.0, 1.0, 0.0)  # Green
        elif self.env._robot_energy > 30:
            glColor3f(1.0, 1.0, 0.0)  # Yellow
        else:
            glColor3f(1.0, 0.0, 0.0)  # Red
        
        # Status light positions around the robot
        for i in range(4):
            angle = i * 90
            light_x = robot_x + 0.18 * math.cos(math.radians(angle))
            light_y = robot_y + 0.18 * math.sin(math.radians(angle))
            self._draw_sphere(light_x, light_y, robot_z + 0.25, 0.02)
    
    def _draw_item(self, x, y, z, item_type):
        """Draw different types of items with unique appearances"""
        color = self._get_item_color(item_type)
        glColor3fv(color)
        
        if item_type == 'electronics':
            # Draw as a small box with circuits
            self._draw_cube(x, y, z, 0.3, 0.3, 0.2)
            glColor3f(0.0, 1.0, 0.0)  # Green circuits
            self._draw_cube(x, y, z + 0.11, 0.25, 0.25, 0.02)
        
        elif item_type == 'clothing':
            # Draw as a soft package
            self._draw_sphere(x, y, z, 0.2)
        
        elif item_type == 'food':
            # Draw as a cylinder (can/package)
            self._draw_cylinder(x, y, z, 0.1, 0.25)
        
        elif item_type == 'books':
            # Draw as stacked rectangles
            for i in range(3):
                self._draw_cube(x, y, z + i * 0.05, 0.25, 0.35, 0.04)
    
    def _get_item_color(self, item_type):
        """Get color for different item types"""
        colors = {
            'electronics': [0.2, 0.2, 0.8],  # Blue
            'clothing': [0.8, 0.2, 0.8],     # Magenta
            'food': [0.8, 0.6, 0.2],         # Orange
            'books': [0.6, 0.4, 0.2],        # Brown
            'unknown': [0.5, 0.5, 0.5]       # Grey
        }
        return colors.get(item_type, colors['unknown'])
    
    def _draw_cube(self, x, y, z, width, height, depth):
        """Draw a 3D cube at specified position"""
        w, h, d = width/2, height/2, depth/2
        
        glBegin(GL_QUADS)
        # Front face
        glVertex3f(x-w, y-h, z-d)
        glVertex3f(x+w, y-h, z-d)
        glVertex3f(x+w, y+h, z-d)
        glVertex3f(x-w, y+h, z-d)
        
        # Back face
        glVertex3f(x-w, y-h, z+d)
        glVertex3f(x-w, y+h, z+d)
        glVertex3f(x+w, y+h, z+d)
        glVertex3f(x+w, y-h, z+d)
        
        # Top face
        glVertex3f(x-w, y+h, z-d)
        glVertex3f(x+w, y+h, z-d)
        glVertex3f(x+w, y+h, z+d)
        glVertex3f(x-w, y+h, z+d)
        
        # Bottom face
        glVertex3f(x-w, y-h, z-d)
        glVertex3f(x-w, y-h, z+d)
        glVertex3f(x+w, y-h, z+d)
        glVertex3f(x+w, y-h, z-d)
        
        # Left face
        glVertex3f(x-w, y-h, z-d)
        glVertex3f(x-w, y+h, z-d)
        glVertex3f(x-w, y+h, z+d)
        glVertex3f(x-w, y-h, z+d)
        
        # Right face
        glVertex3f(x+w, y-h, z-d)
        glVertex3f(x+w, y-h, z+d)
        glVertex3f(x+w, y+h, z+d)
        glVertex3f(x+w, y+h, z-d)
        glEnd()
    
    def _draw_cylinder(self, x, y, z, radius, height):
        """Draw a 3D cylinder"""
        segments = 16
        glBegin(GL_TRIANGLE_FAN)
        # Bottom cap
        glVertex3f(x, y, z - height/2)
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            glVertex3f(x + radius * math.cos(angle), y + radius * math.sin(angle), z - height/2)
        glEnd()
        
        glBegin(GL_TRIANGLE_FAN)
        # Top cap
        glVertex3f(x, y, z + height/2)
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            glVertex3f(x + radius * math.cos(angle), y + radius * math.sin(angle), z + height/2)
        glEnd()
        
        # Side surface
        glBegin(GL_QUAD_STRIP)
        for i in range(segments + 1):
            angle = 2 * math.pi * i / segments
            x_pos = x + radius * math.cos(angle)
            y_pos = y + radius * math.sin(angle)
            glVertex3f(x_pos, y_pos, z - height/2)
            glVertex3f(x_pos, y_pos, z + height/2)
        glEnd()
    
    def _draw_sphere(self, x, y, z, radius):
        """Draw a 3D sphere"""
        glPushMatrix()
        glTranslatef(x, y, z)
        
        # Create sphere using quad strips
        slices = 16
        stacks = 16
        
        for i in range(stacks):
            lat0 = math.pi * (-0.5 + float(i) / stacks)
            z0 = radius * math.sin(lat0)
            zr0 = radius * math.cos(lat0)
            
            lat1 = math.pi * (-0.5 + float(i + 1) / stacks)
            z1 = radius * math.sin(lat1)
            zr1 = radius * math.cos(lat1)
            
            glBegin(GL_QUAD_STRIP)
            for j in range(slices + 1):
                lng = 2 * math.pi * float(j) / slices
                x_pos = math.cos(lng)
                y_pos = math.sin(lng)
                
                glVertex3f(x_pos * zr0, y_pos * zr0, z0)
                glVertex3f(x_pos * zr1, y_pos * zr1, z1)
            glEnd()
        
        glPopMatrix()
    
    def _draw_hud(self):
        """Draw heads-up display with performance metrics"""
        # Switch to 2D rendering for HUD
        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, self.window_size[0], 0, self.window_size[1], -1, 1)
        
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()
        
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_LIGHTING)
        
        # Energy bar
        energy_ratio = self.env._robot_energy / 100
        bar_width = 200
        bar_height = 20
        
        # Background
        glColor3f(0.3, 0.3, 0.3)
        glBegin(GL_QUADS)
        glVertex2f(20, self.window_size[1] - 40)
        glVertex2f(20 + bar_width, self.window_size[1] - 40)
        glVertex2f(20 + bar_width, self.window_size[1] - 40 + bar_height)
        glVertex2f(20, self.window_size[1] - 40 + bar_height)
        glEnd()
        
        # Energy level
        if energy_ratio > 0.6:
            glColor3f(0.2, 0.8, 0.2)  # Green
        elif energy_ratio > 0.3:
            glColor3f(0.8, 0.8, 0.2)  # Yellow
        else:
            glColor3f(0.8, 0.2, 0.2)  # Red
            
        glBegin(GL_QUADS)
        glVertex2f(20, self.window_size[1] - 40)
        glVertex2f(20 + bar_width * energy_ratio, self.window_size[1] - 40)
        glVertex2f(20 + bar_width * energy_ratio, self.window_size[1] - 40 + bar_height)
        glVertex2f(20, self.window_size[1] - 40 + bar_height)
        glEnd()
        
        # Performance indicators (simplified visualization)
        glColor3f(1.0, 1.0, 1.0)
        info = self.env._get_info()
        
        # Tasks completed indicator
        tasks_y = self.window_size[1] - 80
        glBegin(GL_QUADS)
        for i in range(min(info['items_delivered'], 10)):
            x = 20 + i * 25
            glVertex2f(x, tasks_y)
            glVertex2f(x + 20, tasks_y)
            glVertex2f(x + 20, tasks_y + 15)
            glVertex2f(x, tasks_y + 15)
        glEnd()
        
        # Restore 3D rendering
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        
        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)
    
    def handle_input(self):
        """Handle mouse and keyboard input for camera control"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    self.camera_angle_y -= 5
                elif event.key == pygame.K_RIGHT:
                    self.camera_angle_y += 5
                elif event.key == pygame.K_UP:
                    self.camera_angle_x += 5
                elif event.key == pygame.K_DOWN:
                    self.camera_angle_x -= 5
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.camera_distance = max(5, self.camera_distance - 1)
                elif event.key == pygame.K_MINUS:
                    self.camera_distance = min(30, self.camera_distance + 1)
        
        return True
    
    def close(self):
        """Clean up resources"""
        pygame.quit()
