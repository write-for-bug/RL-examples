"""
    自己玩游戏，←对小车施加向左的力，→对小车设置向右的力
    难的很
"""
from itertools import count
import gymnasium as gym
import pygame
# 创建CartPole环境并启用人类渲染模式
env = gym.make("CartPole-v1", render_mode="human")

current_action = 1  # 初始向右移动
step_count = 0
max_steps = 200
done = False
try:
    for i in count():#第i局游戏
        env.reset()
        for t in count():
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    done = True
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        current_action = 0  # 左移
                    elif event.key == pygame.K_RIGHT:
                        current_action = 1  # 右移

            # 根据当前按键选择动作，执行一步环境交互
            observation, reward, terminated, truncated, info = env.step(current_action)
            done = terminated or truncated
            step_count += 1
            # 检测游戏是否结束
            if done:
                print(f"第{i}局游戏在第 {step_count} 步结束")
                step_count = 0
                break
finally:
# 关闭环境以释放资源
    env.close()