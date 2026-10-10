import gymnasium as gym

def main():
    env = gym.make("CartPole-v1", render_mode="human")
    print("Observation space:", env.observation_space)
    print("Observation: cart position, cart velocity, pole angle, pole angular velocity")
    print("Action space:", env.action_space, "(0=left, 1=right)")
    obs, _ = env.reset()
    done = False
    while not done:
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
    env.close()

if __name__ == "__main__": main()
