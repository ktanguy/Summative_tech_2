# AI Warehouse Automation - Live Demo

**Reinforcement Learning Project with Live Robot Animation**

## Overview

This project demonstrates reinforcement learning algorithms in a warehouse automation scenario with real-time visualization. The system shows multiple RL agents (DQN and PPO) learning to perform warehouse tasks including item pickup, delivery, and energy management.

## Quick Start

### Main Demo
```bash
python3 live_robot_demo.py
```
This runs the main demonstration showing robots moving and learning in real-time.

### Training Models
```bash
python3 training/dqn_training.py       # Train DQN agent
python3 training/pg_training.py        # Train PPO, A2C, REINFORCE agents
```

### Web Demo
```bash
open professional_demo.html
```

## Project Structure
```
live_robot_demo.py           # Main demonstration script
environment/                 # Custom warehouse environment
training/                    # RL algorithm implementations
models/                      # Trained model weights
results/                     # Training results and metrics
professional_demo.html       # Web-based demonstration
```

## Features

- **Multi-Agent Learning**: Red Robot (DQN) and Blue Robot (PPO) with different learning strategies
- **Real-time Visualization**: Live charts showing learning progress and performance metrics
- **Energy Management**: Robots autonomously manage battery levels and charging behavior
- **Task Completion**: Dynamic pickup and delivery of warehouse items
- **Algorithm Comparison**: Implementation of DQN, PPO, A2C, and REINFORCE algorithms

## Technical Details

The project implements four reinforcement learning algorithms in a custom warehouse environment with live visualization capabilities. The system is optimized for educational demonstrations and research purposes.
