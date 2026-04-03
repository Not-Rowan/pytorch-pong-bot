import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from collections import deque
import random
import numpy as np


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        # nn.Linear is the pytorch version of a fully connected layer
        self.fc1 = nn.Linear(n_observations, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, x):
        # x is a batch of states: [batch size, 4]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x) # returns linear q-values for each action: [batch_size, 2]

# select device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Metal (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU")


######################
## Helper Functions ##
def select_action(state):
    global steps_done
    sample = random.random()

    epsilon_value = max(EPS_END, EPS_START - (((EPS_START - EPS_END) / EPS_DECAY) * steps_done))

    if sample < epsilon_value and train_or_load == "t":
        # choose random action (0 for down, 1 for up, 0.5 for no move)
        return random.choice([0, 0.5, 1])
    else:
        # have online network select best action
        with torch.no_grad():
            state = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0) # format into tensor for network
            return online_net(state).argmax().cpu().item() # move to cpu then return action
        
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    # sample SARS batches from memory
    batch = random.sample(memory, BATCH_SIZE)
    states, actions, rewards, next_states, dones = zip(*batch)

    # convert into pytorch tensors
    state_batch = torch.as_tensor(np.array(states), dtype=torch.float32, device=device)
    action_batch = torch.as_tensor(actions, dtype=torch.long, device=device).unsqueeze(dim=1)
    reward_batch = torch.as_tensor(rewards, dtype=torch.float32, device=device).unsqueeze(dim=1)
    next_state_batch = torch.as_tensor(np.array(next_states), dtype=torch.float32, device=device)
    done_batch = torch.as_tensor(dones, dtype=torch.long, device=device).unsqueeze(dim=1)


    with torch.no_grad():
        # next best action according to online net
        next_best_actions = online_net(next_state_batch).argmax(dim=1, keepdims=True)

        # compute Q value for each action and select the best action out of these Q values according to the online net
        target_action_vals = target_net(next_state_batch).gather(1, next_best_actions)

    # compute the target Q values then use this to update the online network
    target_Q_vals = reward_batch + GAMMA * target_action_vals * (1 - done_batch)
    current_Q_vals = online_net(state_batch).gather(1, action_batch) # get current Q-values for the actions we actually took

    # compute loss with MSE
    loss = F.mse_loss(current_Q_vals, target_Q_vals)

    # calculate and update gradients
    optimizer.zero_grad() # zero gradients
    loss.backward() # calculate gradients (backprop)
    optimizer.step() # apply gradients
    


def load_and_inference():
    online_net.load_state_dict(torch.load(PATH+"online_net.pth"))
    online_net.eval() # set model to eval mode

    while True:
        state, _ = env.reset()
        total_reward = 0
        action = select_action(state)

        while True:
            # take action in env
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

            next_action = select_action(next_state)

            state = next_state
            action = next_action

            if done:
                print(f"total reward = {total_reward}")
                break
        

def train_model():
    global steps_done

    # make target a copy of online
    target_net.load_state_dict(online_net.state_dict())

    for episode_num in range(EPISODES):
        state, _ = env.reset() # set env to initial state
        total_reward = 0
        action = select_action(state) # select initial action

        for _ in range(EPISODE_LEN):
            # take action in env
            next_state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward # increment total reward
            done = terminated or truncated # handle end-state

            # select next action for the new state
            next_action = select_action(next_state)

            # append SARS and done flag to queue
            memory.append([state, action, reward, next_state, done])

            state = next_state # move to next state
            action = next_action # move to next action
            
            optimize_model()

            steps_done += 1

            # update target network after TARGET_UPDATE_FREQ steps
            if steps_done % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(online_net.state_dict())

            if done:
                print(f"Episode {episode_num}: total reward = {total_reward}")
                break

    # save model and close window when done
    print("training complete :D")
    torch.save(online_net.state_dict(), PATH+"online_net.pth")
    torch.save(target_net.state_dict(), PATH+"target_net.pth")
    env.close()


# Hyperparameters
LR = 1e-3
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
BATCH_SIZE = 32
MEMORY_SIZE = 10000
TARGET_UPDATE_FREQ = 30

EPISODE_LEN = 1000
EPISODES = 200

# misc
PATH = "./"
RENDER_MODE = "human" # usually "human" or None

random.seed()


# initialize env, online net, target net, and optimizer
env = gym.make("CartPole-v1", render_mode=RENDER_MODE)
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.n

online_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
optimizer = torch.optim.AdamW(online_net.parameters(), lr=LR, amsgrad=True)

# misc
steps_done = 0


train_or_load = input("Would you like to load a model or train from scratch? (t/l): ")
if train_or_load == 't':
    # misc
    memory = deque(maxlen=MEMORY_SIZE)

    train_model()
elif train_or_load == 'l':
    load_and_inference()