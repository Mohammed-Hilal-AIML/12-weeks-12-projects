# Week 11 — Autonomous Game-Playing Agent with Reinforcement Learning

Train a Deep Q-Network (DQN) on Gymnasium's `CartPole-v1` environment, monitor rewards, and evaluate the saved agent. This is an educational project; report only results you actually measure.

## What you'll learn
- RL concepts: state, action, reward, agent, environment
- Q-learning and Bellman TD targets
- DQN, replay buffer, target network, epsilon-greedy exploration
- Gradient clipping, reward plots, and evaluation

## Structure
```text
Week_11_Autonomous_Game_Playing_Agent_Reinforcement_Learning/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   ├── baseline.py
│   ├── inspect_environment.py
│   ├── train_dqn.py
│   └── evaluate.py
├── data/README.md
├── models/.gitkeep
└── outputs/README.md
```

## 1. Install (Windows / Python 3.12)
Open PowerShell in the repository root:
```powershell
py -3.12 --version
py -3.12 -m pip install --upgrade pip
py -3.12 -m pip install -r requirements.txt
```
If PyTorch installation fails, use the official PyTorch website's install selector for your CPU/CUDA configuration.

## 2. Run random baseline
```powershell
py -3.12 scripts\baseline.py --episodes 100
```

## 3. Inspect CartPole
```powershell
py -3.12 scripts\inspect_environment.py
```
Observation contains four values (cart position, cart velocity, pole angle, pole angular velocity). Actions are `0` (left) and `1` (right). A window opens for a random episode.

## 4. Train DQN
```powershell
py -3.12 scripts\train_dqn.py --episodes 500
```
Settings follow the project manual: Q-network `4 → 128 → 128 → 2`; replay capacity 10,000; batch size 64; epsilon from 1.0 down to 0.01; target network sync every 100 environment steps; MSE loss and gradient clipping. The script saves `models/dqn_cartpole.pt`, `outputs/training_rewards.csv`, and `outputs/training_rewards.png`.

Training may take a while on CPU. CartPole is considered solved for this project when the 100-episode rolling average is at least 195. The script prints measured rewards; do not claim success unless your run reaches the threshold.

## 5. Evaluate
```powershell
py -3.12 scripts\evaluate.py --episodes 20
```
Train first so the checkpoint exists.

## Troubleshooting
- `can't open file`: run from the project root and include `scripts\` in the command.
- Slow training: check CPU usage in Task Manager; don't assume it's frozen if episode logs continue.
- Missing model: run training successfully before evaluation.
- No dataset is required; Gymnasium provides the environment.

## GitHub
Suggested repository name: `Week_11_Autonomous_Game_Playing_Agent_Reinforcement_Learning`

Suggested description: `Deep Q-Network (DQN) agent for CartPole with experience replay, target network, epsilon-greedy exploration, reward monitoring, and evaluation.`

References: https://gymnasium.farama.org/ · https://pytorch.org/


## Included pretrained checkpoint

A starter checkpoint is included at `models/dqn_cartpole.pt`. It is a trained DQN checkpoint and is compatible with the `QNetwork` architecture in `scripts/evaluate.py`. It is provided as a starting point, not as a claim that CartPole has been solved. Re-evaluate it on your installed Gymnasium version; if its mean reward is below 195, run `py -3.12 scripts\train_dqn.py --episodes 500` to train a fresh checkpoint locally.
