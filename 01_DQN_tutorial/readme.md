# 01_DQN_tutorial
## 简介
    一个简易的DQN训练和可视化实现，来自官方教程略改，初入rl练手项目
## 新名词
    policynet和targetnet
    ε-greedy策略
## 核心思想
    智能体根据policynet选择自己的“最优策略”（ε-greedy策略下支持新的探索）
    targetnet是稳定回报，用于计算Q(s')，长期回报
    用policynet缓步更新targetnet
