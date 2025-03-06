import pygame
import gymnasium as gym
from itertools import count

# 创建CartPole环境
env_render = gym.make("CartPole-v1", render_mode = 'human')
env_no_render = gym.make("CartPole-v1")
try:
    # 初始化环境状态
    for i in count():
        if i%100 ==0:
            env = env_render
        else:
            env = env_no_render
        # 运行游戏循环（例如200个时间步）
        _,_ = env.reset()
        for t in count(0):
            # 随机选择左右移动动作（0: 左, 1: 右）
            #env.render()
            action = env.action_space.sample()
            # 执行动作并获取新状态、奖励等信息
            observation, reward, done, info ,_ = env.step(action)

            # 检测游戏是否结束
            if done:
                print(f"第{i}局游戏结束于第 {t + 1} 步")
                break
finally:
    # 关闭环境释放资源
    env_render.close()
    env_no_render.close()