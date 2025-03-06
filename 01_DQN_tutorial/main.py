import gymnasium as gym
import math
from matplotlib import pyplot as plt
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
from dqn import DQN
from memory import ReplayMemory,Transition
import random
env1 = gym.make("CartPole-v1", render_mode="human")
env2 = gym.make("CartPole-v1")

plt.ion()#开启交互模式：允许图形实时更新（例如在循环中动态添加数据时）

# if GPU is to be used
device = 'gpu' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 128    #sample num from replay buffer
GAMMA = 0.99        #long return factor
EPS_START = 0.9     #starting value of epsilon
EPS_END = 0.05      #final value of epsilon
EPS_DECAY = 1000    #rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005         #update rate of the target network
LR = 1e-3           #learning rate of the ``AdamW`` optimizer


n_actions = env1.action_space.n      # number of actions from gym action space
# state, info = env1.reset()           #number of state observations
# n_observations = len(state)
n_observations = 4
policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    '''以ε-greedy策略选择action'''
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)


episode_durations = []

plt.figure(1)
def plot_durations(show_result=False):

    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    #state, next_state, action, reward
    batch = Transition(*zip(*transitions))
    '''
    map(function, iterable, ...)用于将一个指定的函数应用于可迭代对象（如列表、元组等）的每一个元素，并返回一个包含结果的迭代器
    '''
    # batch.next_state可能为None，下面两行是为了生成掩码把next_state为None的sample给剔除
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)


    '''
        用policy_net计算Q(s),在s状态下每个action的Q值，action_batch记录每个sample选择的action
        .gather筛选出Q(s)中实际选择的操作的Q值
        state_action_values就是实际的action的Q值
    '''
    Q_s = policy_net(state_batch)
    state_action_values = Q_s.gather(1, action_batch)

    '''
        计算V(s')            V(s') = max Q(s' , a)
        target_net计算Q(s'),包含next_state下对应每个action的Q值，最大的就是V(s')
        ps: next_state要剔除结束的为None的sample,非none sample经过targetnet，none sample直接设为0
    '''
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        Q_next_s = target_net(non_final_next_states)
        next_state_values[non_final_mask] = Q_next_s.max(1).values
    # Compute the expected Q values
    # r + γV(s')
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    #torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


num_episodes = 2000
env = None
for i_episode in range(num_episodes):
    # env1是训练环境，env2是渲染环境，好像初始化设置了渲染模式后续无法改变，所以设置两个环境，训练不可视化，每100个episode展示一次训练成果
    if i_episode % 20 == 0:
        env = env1
    else:
        env = env2
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)#转成tensor
    for t in count():
        action = select_action(state)
        observation, reward, terminated, truncated, info = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            if i_episode % 100 == 0:
                plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()
