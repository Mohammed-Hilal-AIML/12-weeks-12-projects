import argparse, random
from collections import deque
from pathlib import Path
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.optim import Adam

ROOT = Path(__file__).resolve().parents[1]

class ReplayBuffer:
    def __init__(self, capacity=10000): self.buffer = deque(maxlen=capacity)
    def add(self, *transition): self.buffer.append(transition)
    def sample(self, batch_size, device):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (torch.tensor(np.asarray(s), dtype=torch.float32, device=device),
                torch.tensor(a, dtype=torch.long, device=device).unsqueeze(1),
                torch.tensor(r, dtype=torch.float32, device=device).unsqueeze(1),
                torch.tensor(np.asarray(ns), dtype=torch.float32, device=device),
                torch.tensor(d, dtype=torch.float32, device=device).unsqueeze(1))
    def __len__(self): return len(self.buffer)

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(4,128), nn.ReLU(),
                                 nn.Linear(128,128), nn.ReLU(),
                                 nn.Linear(128,2))
    def forward(self, x): return self.net(x)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}; episodes: {args.episodes}", flush=True)
    env = gym.make("CartPole-v1")
    online, target = QNetwork().to(device), QNetwork().to(device)
    target.load_state_dict(online.state_dict()); target.eval()
    optimizer, loss_fn = Adam(online.parameters(), lr=1e-3), nn.MSELoss()
    memory = ReplayBuffer(10000)
    batch_size, gamma = 64, 0.99
    epsilon, epsilon_min, epsilon_decay = 1.0, 0.01, 0.995
    step, rewards, rolling = 0, [], deque(maxlen=100)
    (ROOT/"models").mkdir(exist_ok=True); (ROOT/"outputs").mkdir(exist_ok=True)
    for ep in range(1, args.episodes+1):
        state, _ = env.reset(seed=args.seed+ep)
        total, done = 0.0, False
        while not done:
            step += 1
            if random.random() < epsilon: action = env.action_space.sample()
            else:
                with torch.no_grad():
                    action = int(online(torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)).argmax(1).item())
            ns, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            memory.add(state, action, reward, ns, done)
            state, total = ns, total + reward
            if len(memory) >= batch_size:
                s,a,r,n_s,d = memory.sample(batch_size, device)
                q = online(s).gather(1,a)
                with torch.no_grad(): target_q = r + gamma * target(n_s).max(1,keepdim=True).values * (1-d)
                loss = loss_fn(q,target_q)
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(online.parameters(), 10.0)
                optimizer.step()
            if step % 100 == 0: target.load_state_dict(online.state_dict())
        epsilon = max(epsilon_min, epsilon*epsilon_decay)
        rewards.append(total); rolling.append(total)
        if ep == 1 or ep % 10 == 0:
            avg = float(np.mean(rolling))
            print(f"Episode {ep}/{args.episodes} | reward={total:.0f} | rolling100={avg:.2f} | epsilon={epsilon:.3f}", flush=True)
            if len(rolling) == 100 and avg >= 195: print(f"Solved threshold reached at episode {ep}.", flush=True)
    torch.save(online.state_dict(), ROOT/"models"/"dqn_cartpole.pt")
    np.savetxt(ROOT/"outputs"/"training_rewards.csv", rewards, delimiter=",", header="episode_reward", comments="")
    averages = [np.mean(rewards[max(0,i-99):i+1]) for i in range(len(rewards))]
    plt.figure(figsize=(10,5)); plt.plot(rewards, alpha=.4, label="Episode reward")
    plt.plot(averages, label="Rolling average (up to 100)")
    plt.axhline(195, linestyle="--", label="Solved threshold (195)")
    plt.xlabel("Episode"); plt.ylabel("Reward"); plt.title("DQN on CartPole-v1")
    plt.legend(); plt.tight_layout(); plt.savefig(ROOT/"outputs"/"training_rewards.png", dpi=150)
    env.close()
    print("Saved models/dqn_cartpole.pt and outputs/training_rewards.png", flush=True)

if __name__ == "__main__": main()
