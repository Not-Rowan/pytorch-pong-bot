import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import numpy as np
import socket
import time

PORT = 1235

WIDTH = 800
HEIGHT = 600
BUFFER_SIZE = 1024

PATH = "./"

# Hyperparameters
LR = 1e-5
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_STEPS = 2_000_000
BATCH_SIZE = 32
MEMORY_SIZE = 100000
TARGET_UPDATE_FREQ = 10000

EPISODE_LEN = 5000
EPISODES = 10000

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
'''if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Metal (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU")'''

# network is small enough for the cpu
device = torch.device("cpu")



######################
## Helper Functions ##
def normalize_value(value, min_val, max_val):
    if max_val == min_val:
        return 0.0
    return 2 * ((value-min_val) / (max_val-min_val)) - 1

def recv_game_state():
    data = conn.recv(BUFFER_SIZE).decode("utf-8")
    state = list(map(float, data.strip("{}").split(",")))
    return state

def translate_state(state):
    # get rid of points and normalize values from their respective max and min values to -1 and 1 (this helps the network learn faster)
    player_paddle_y, opponent_paddle_y, _, _, ball_x, ball_y, ball_velocity_x, ball_velocity_y = state
    player_paddle_y = normalize_value(player_paddle_y, 0, HEIGHT)
    opponent_paddle_y = normalize_value(opponent_paddle_y, 0, HEIGHT)
    ball_x = normalize_value(ball_x, 0, WIDTH)
    ball_y = normalize_value(ball_y, 0, HEIGHT)
    ball_velocity_x = normalize_value(ball_velocity_x, -5, 5)
    ball_velocity_y = normalize_value(ball_velocity_y, -5, 5)
    return [player_paddle_y, opponent_paddle_y, ball_x, ball_y, ball_velocity_x, ball_velocity_y]

def send_move(move):
    # convert the move indices to values the game can understand
    # 0 for down, 1 for up, 0.5 for no move (this maps to 0, 1, 2 respectively)
    if move == 2:
        converted_move = 0.5
    else:
        converted_move = float(move)
    formatted_move = "{" + f"{converted_move:.1f}" + "}"
    conn.sendall(formatted_move.encode("utf-8"))

def calculate_reward(current_state, next_state, current_action, next_action):
    reward = 0

    # unpack both state lists
    player_paddle_y, opponent_paddle_y, player_points, opponent_points, ball_x, ball_y, ball_velocity_x, ball_velocity_y = current_state
    next_player_paddle_y, next_opponent_paddle_y, next_player_points, next_opponent_points, next_ball_x, next_ball_y, next_ball_velocity_x, next_ball_velocity_y = next_state

    # reward the network if it is on top of the ball or moves in the direction of the ball. otherwise punish
    '''if ball_y >= player_paddle_y and ball_y <= (player_paddle_y + 100):
        # on top of ball (if (ballY >= networkPaddleY || ballY <= networkPaddleY+100))
        reward += 0.15
    elif (player_paddle_y > ball_y and current_action == 1.0) or ((player_paddle_y + 100) < ball_y and current_action == 0):
        # moving towards the ball (if ((networkPaddleY > ballY && move == up) || ((networkPaddleY+100) < ballY && move == down)))
        reward += 0.1
    else:
        reward += -0.1'''

    # reward the network if the network has scored a point
    if next_player_points > 0 or next_opponent_points > 0:
        if (next_player_points != player_points) or (next_opponent_points != opponent_points):
            if next_player_points > player_points:
                reward += 3.0
            elif next_opponent_points > opponent_points:
                reward -= 3.0

    return reward


def select_action(state):
    global steps_done
    global epsilon_value
    sample = random.random()

    if sample < epsilon_value and train_or_load == "t":
        # choose random action (0 for down, 1 for up, 2 for no move)
        return random.choice([0, 1, 2])
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
    done_batch = torch.as_tensor(dones, dtype=torch.float32, device=device).unsqueeze(dim=1)

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

def train_model():
    global steps_done
    global epsilon_value

    # make target a copy of online
    target_net.load_state_dict(online_net.state_dict())

    for episode_num in range(EPISODES):
        if episode_num == 0:
            state = recv_game_state() # get initial game state
            translated_state = translate_state(state)
            action = select_action(translated_state) # select initial action
        total_reward = 0
        done = 0

        for episode_steps in range(EPISODE_LEN):
            # take action in env & recv new state
            send_move(action)
            next_state = recv_game_state()
            while next_state == state:
                time.sleep(0.001)
                next_state = recv_game_state()
            translated_next_state = translate_state(next_state)

            # select next action for the new state
            next_action = select_action(translated_next_state)

            # get rewards
            reward = calculate_reward(state, next_state, action, next_action)
            total_reward += reward # increment total reward counter

            # set endGame flag to 1 if a point has been scored only if points are greater than 0
            # of if we reached EPISODE_LEN
            _, _, player_points, opponent_points, _, _, _, _ = state
            _, _, next_player_points, next_opponent_points, _, _, _, _ = next_state
            if (next_player_points != player_points) or (next_opponent_points != opponent_points) or (episode_steps == EPISODE_LEN-1):
                done = 1
                print(f"Episode Steps: {episode_steps}")

            # append SARS and done to queue
            memory.append([translated_state, action, reward, translated_next_state, done])

            state = next_state # move to next state
            translated_state = translated_next_state 
            action = next_action # move to next action
            
            optimize_model()

            steps_done += 1

            # update target network after TARGET_UPDATE_FREQ steps
            if steps_done % TARGET_UPDATE_FREQ == 0:
                target_net.load_state_dict(online_net.state_dict())

            # calculate new epsilon value
            if epsilon_value > EPS_END:
                epsilon_value = max(EPS_END, epsilon_value - ((EPS_START-EPS_END) / EPS_DECAY_STEPS))

            if done:
                break

        print(f"Episode {episode_num}: total reward = {total_reward}")
        print(f"epsilon value: {epsilon_value}")


    # save model and close window when done
    print("training complete :D")
    torch.save(online_net.state_dict(), PATH+"online_net.pth")
    torch.save(target_net.state_dict(), PATH+"target_net.pth")

def load_and_inference():
    online_net.load_state_dict(torch.load(PATH+"online_net.pth"))
    online_net.eval() # set model to eval mode

    state = recv_game_state() # get initial game state
    translated_state = translate_state(state)
    action = select_action(translated_state) # select initial action

    while True:
        # take action in env & recv new state
        send_move(action)
        next_state = recv_game_state()
        translated_next_state = translate_state(next_state)

        # select next action for the new state
        next_action = select_action(translated_next_state)

        state = next_state # move to next state
        action = next_action # move to next action


random.seed()

# ask first (i don't wanna run the game first then switch back here)
train_or_load = input("Would you like to load a model or train from scratch? (t/l): ")

# socket connection stuff
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", PORT))
print(f"Binded to port {PORT} successfully. Listening for game client (run the game bud)")
s.listen()
conn, addr = s.accept()

print(f"Connected to client successfully ({addr})")

gameConstants = "{" + str(WIDTH) + ", " + str(HEIGHT) + "}"
conn.sendall(gameConstants.encode("utf-8"))


# Initialize the neural network
n_observations = 6
n_actions = 3

online_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
#online_net = torch.compile(online_net, mode="reduce-overhead").to(device)
#target_net = torch.compile(target_net, mode="reduce-overhead").to(device)
optimizer = torch.optim.AdamW(online_net.parameters(), lr=LR)

steps_done = 0
epsilon_value = EPS_START

if train_or_load == 't':
    # misc
    memory = deque(maxlen=MEMORY_SIZE)

    train_model()
elif train_or_load == 'l':
    load_and_inference()