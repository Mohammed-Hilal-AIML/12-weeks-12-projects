import argparse
import numpy as np
import gymnasium as gym

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    args = parser.parse_args()
    env = gym.make("CartPole-v1")
    rewards = []
    for ep in range(1, args.episodes + 1):
        obs, _ = env.reset()
        total, done = 0.0, False
        while not done:
            obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
            total += reward
            done = terminated or truncated
        rewards.append(total)
        if ep % 10 == 0:
            print(f"Episode {ep}/{args.episodes} | last-10 average={np.mean(rewards[-10:]):.2f}")
    print(f"Random baseline average reward: {np.mean(rewards):.2f}")
    env.close()

if __name__ == "__main__": main()
