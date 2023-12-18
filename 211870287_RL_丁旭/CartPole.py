import gym
import numpy as np
import torch
import torch.optim as optim
from matplotlib import animation
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F


import random
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'next_state', 'action', 'reward', 'mask'))


class Memory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.capacity = capacity

    def push(self, state, next_state, action, reward, mask):
        self.memory.append(Transition(state, next_state, action, reward, mask))

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return batch

    def __len__(self):
        return len(self.memory)

class QNet(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(QNet, self).__init__()
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs

        self.fc1 = nn.Linear(num_inputs, 128)
        self.fc2 = nn.Linear(128, num_outputs)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform(m.weight)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        qvalue = self.fc2(x)
        return qvalue

    @classmethod
    def train_model(cls, evaluate_net, target_net, optimizer, batch):
        states = torch.stack(batch.state)
        next_states = torch.stack(batch.next_state)
        actions = torch.Tensor(batch.action).float()
        rewards = torch.Tensor(batch.reward)
        masks = torch.Tensor(batch.mask)

        pred = evaluate_net(states).squeeze(1)
        next_pred = target_net(next_states).squeeze(1)

        pred = torch.sum(pred.mul(actions), dim=1)

        target = rewards + masks * gamma * next_pred.max(1)[0]

        loss = F.mse_loss(pred, target.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss

    def get_action(self, input):
        qvalue = self.forward(input)
        # action为每一行最大值的索引
        _, action = torch.max(qvalue, 1)
        return action.numpy()[0]


env_name = 'CartPole-v1'
gamma = 0.99
batch_size = 32
lr = 0.001
initial_exploration = 1000
goal_score = 250
log_interval = 10
update_target = 100
replay_memory_capacity = 1000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_action(state, target_net, epsilon, env):
    """
    若生成的随机数小于epsilon，则随机选择一个动作
    否则使用target_net(state)输出的结果
    :return:下一步采取的动作,值为0或1
    """
    if np.random.rand() <= epsilon:
        return env.action_space.sample()
    else:
        return target_net.get_action(state)


def update_target_model(evaluate_net, target_net):
    """
    将online_net中的weights赋值给target_net
    :param evaluate_net: 权重来源的网络
    :param target_net: 需要更新参数的网络
    :return: no return value
    """
    target_net.load_state_dict(evaluate_net.state_dict())

def display_frames_as_gif(frames):
    """
    将frames中的每一帧保存为gif图
    :param frames: 一个数组 保存游戏的每一帧
    :return: no return value
    """
    patch = plt.imshow(frames[0])
    plt.axis("off")

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval = 5)
    anim.save("./CartPole_v1_result.gif", writer="pillow", fps = 30)



def main():
    """
    使用DQN算法训练网络
    :return: no return value
    """
    env = gym.make(env_name,render_mode='rgb_array')
    torch.manual_seed(500)

    # 获取观察空间的shape和动作空间的shape
    num_inputs = env.observation_space.shape[0]
    num_actions = env.action_space.n
    print('state size:', num_inputs)
    print('action size:', num_actions)

    # 创建evaluate_net 和 target_net
    evaluate_net = QNet(num_inputs, num_actions)
    target_net = QNet(num_inputs, num_actions)
    update_target_model(evaluate_net, target_net)

    optimizer = optim.Adam(evaluate_net.parameters(), lr=lr)
    writer = SummaryWriter('logs')

    evaluate_net.to(device)
    target_net.to(device)
    evaluate_net.train()
    target_net.train()
    memory = Memory(replay_memory_capacity)
    running_score = 0
    steps = 0
    epsilon = 1.0
    loss = 0

    escape = False
    frames = []
    # 最多训练3000个episode
    for e in range(3000):
        done = False

        score = 0
        state = env.reset()[0]
        state = torch.Tensor(state).to(device)
        state = state.unsqueeze(0)

        # 开始一次episode,即一次游戏
        while not done:
            steps += 1
            action = get_action(state, target_net, epsilon, env)
            next_state, reward, done = env.step(action)[:3]
            # 如果应该要退出训练 将最后一次游戏保存下来
            if escape:
                frames.append(env.render())

            next_state = torch.Tensor(next_state).unsqueeze(0)

            mask = 0 if done else 1
            reward = reward if not done or score == 499 else -1
            # 将action独热编码
            action_one_hot = np.zeros(2)
            action_one_hot[action] = 1
            memory.push(state, next_state, action_one_hot, reward, mask)

            score += reward
            state = next_state

            # 经验回放 Experience Replay
            # steps<=initial_exploration(1000)的时候仅将样本存入记忆库memory
            # 之后当样本量足够之后才开始训练过程
            if steps > initial_exploration:
                epsilon -= 0.00005
                epsilon = max(epsilon, 0.1)

                batch = memory.sample(batch_size)
                loss = QNet.train_model(evaluate_net, target_net, optimizer, batch)

                if steps % update_target == 0:
                    update_target_model(evaluate_net, target_net)
        if escape:
            env.close()
            display_frames_as_gif(frames)
            break

        score = score if score == 500.0 else score + 1
        # running_score是一个平滑的分数指标，用于跟踪每个episode中的平均表现。
        # 通过将running_score更新为过去分数和本轮分数的加权平均值，可以减少分数的波动性，使其更具稳定性。
        running_score = 0.99 * running_score + 0.01 * score
        if e % log_interval == 0:
            print('{} episode | score: {:.2f} | epsilon: {:.2f}'.format(
                e, running_score, epsilon))
            writer.add_scalar('log/score', float(running_score), e)
            writer.add_scalar('log/loss', float(loss), e)

        if running_score > goal_score:
            escape = True
    env.close()


if __name__ == "__main__":
    main()
