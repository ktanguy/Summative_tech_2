# AI Warehouse Automation - Live Demo
🤖 **Reinforcement Learning Project with Live Robot Animation**

## 🎯 What This Does
Shows 2 robots (DQN & PPO) learning to work in a warehouse in **real-time animation**!

## 🚀 How to Run (Super Simple!)

### Main Demo (For Presentations):
```bash
python3 live_robot_demo.py
```
**This is your main demo!** - Shows robots moving and learning live

### Train Models:
```bash
python3 training/dqn_training.py       # Train DQN
python3 training/pg_training.py        # Train PPO, A2C, REINFORCE  
```

### Web Demo:
```bash
open professional_demo.html
```

## 📁 What's in Here
```
📱 live_robot_demo.py        ← MAIN DEMO (run this!)
🏗️ environment/              ← Custom warehouse 
🤖 training/                 ← 4 RL algorithms
🧠 models/                   ← Trained models
📊 results/                  ← Training results
🌐 professional_demo.html    ← Web demo
📋 PRESENTATION_GUIDE.md     ← How to present
```

## 🎬 What You'll See
- 🔴 **Red Robot (DQN)** - Learns one strategy
- 🔵 **Blue Robot (PPO)** - Learns different strategy  
- 📈 **Live charts** showing learning progress
- ⚡ **Energy management** (robots go charge)
- ✅ **Task completion** (pick up & deliver items)

## 🎓 For Your Assignment
✅ Custom environment  
✅ 4 RL algorithms (DQN, PPO, A2C, REINFORCE)  
✅ Live visualization  
✅ Professional presentation  
✅ Clean code structure  

**Perfect for summative assessments!**

---
*MacBook Pro 2018 compatible - No Unity needed!*
