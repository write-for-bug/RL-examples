import gymnasium as gym
for i in gym.envs.registry.keys():
    print(i)
print(len(gym.envs.registry))