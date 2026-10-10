
import argparse
from pathlib import Path

import numpy as np
import gymnasium as gym
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the trained DQN CartPole agent"
    )
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    checkpoint = ROOT / "models" / "dqn_cartpole.pt"

    if not checkpoint.exists():
        raise SystemExit(
            f"Model not found: {checkpoint}\n"
            "Train the agent first using scripts\\train_dqn.py"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = QNetwork().to(device)
    model.load_state_dict(
        torch.load(
            checkpoint,
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    rewards = []

    env = gym.make("CartPole-v1")

    print("\n" + "=" * 48)
    print("       DQN CARTPOLE EVALUATION")
    print("=" * 48)
    print(f"Device             : {device}")
    print(f"Episodes requested : {args.episodes}")
    print(f"Model              : {checkpoint.name}")
    print("-" * 48)

    try:
        for ep in range(1, args.episodes + 1):
            state, _ = env.reset(seed=10000 + ep)
            total_reward = 0
            done = False

            while not done:
                with torch.no_grad():
                    state_tensor = torch.tensor(
                        state,
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0)

                    action = int(
                        model(state_tensor).argmax(dim=1).item()
                    )

                state, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward
                done = terminated or truncated

            rewards.append(total_reward)
            print(
                f"Episode {ep:>3}/{args.episodes}: "
                f"score = {total_reward:>3}"
            )
    finally:
        env.close()

    scores = np.asarray(rewards, dtype=float)
    mean_score = float(np.mean(scores))

    print("-" * 48)
    print("             FINAL RESULTS")
    print("-" * 48)
    print(f"Episodes evaluated : {len(scores)}")
    print(f"Mean score         : {mean_score:.2f}")
    print(f"Best score         : {np.max(scores):.0f}")
    print(f"Worst score        : {np.min(scores):.0f}")
    print(f"Standard deviation : {np.std(scores):.2f}")
    print(f"Scores >= 195      : {np.sum(scores >= 195)}"
          f" / {len(scores)}")

    print("-" * 48)

    if mean_score >= 195:
        print("Result: Mean evaluation score reached 195.")
    else:
        print("Result: Agent has not reached a mean score of 195 yet.")

    print("=" * 48)
    print(
        "Note: The manual's official target is a rolling "
        "average >= 195 over 100 training episodes."
    )


if __name__ == "__main__":
    main()
