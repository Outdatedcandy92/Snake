import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point
from model import Linear_QNet, QTrainer
from plot import plot, save
import time
import sys
import signal

MAX_MEMORY = 300_000
BATCH_SIZE = 3000
LR = 0.001
EPSILON = 0
DR = 0.9
HIDDEN_LAYER = 256

MODE = 1

origt =  time.time()



class Agent:

    def __init__(self):
        self.n_games = 0
        self.epsilon = EPSILON # randomness
        self.gamma = DR # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, HIDDEN_LAYER, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, game):
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
    


# Your training code here

# Your training code here
def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()
    def signal_handler(sig, frame):
        print('Saving model...')
        agent.model.save()
        time.sleep(0.2)
        print('Model saved')
        time.sleep(0.2)
        save()
        print('Plot saved as plot.png')
        time.sleep(0.2)
        print('Total Time:',time.time()-origt)
        print(f'Settings: \n MAX_MEMORY: {MAX_MEMORY} \n BATCH_SIZE: {BATCH_SIZE} \n LR: {LR} \n EPSILON: {EPSILON} \n DR: {DR} \n HIDDEN_LAYER: {HIDDEN_LAYER} \n MODE: {MODE} \n')


        sys.exit(0)
        
        

  
    while MODE==1:
        signal.signal(signal.SIGINT, signal_handler)
        start_t = time.time()
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()
            current_t = time.time()
            

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            print('Game', agent.n_games, 'Score',score, 'Record:', record, 'Mean Score:', mean_score,'Time:', round(current_t-start_t, 4))
    
     # Set the model to evaluation mode

    #agent.load_model("model.pth")
    
    while MODE==2:  # Or some condition to stop the game after N episodes
        # Get the current state
        state_old = agent.get_state(game)
        
        # Get move from the pre-trained model
        final_move = agent.get_action(state_old)
        
        # Perform move and get new state
        reward, done, score = game.play_step(final_move)
        
        if done:
            # Reset the game to start a new episode
            game.reset()
            agent.n_games += 1
            
            # Optionally update records and print scores
            if score > record:
                record = score  # Assuming 'record' is defined elsewhere
            plot_scores.append(score)  # Assuming 'plot_scores' is defined elsewhere
            total_score += score  # Assuming 'total_score' is defined elsewhere
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)  # Assuming 'plot_mean_scores' is defined elsewhere
            plot(plot_scores, plot_mean_scores)  # Assuming a function 'plot' is defined elsewhere
            print(f'Game {agent.n_games}, Score: {score}, Record: {record}, Mean Score: {mean_score}')

if __name__ == '__main__':
    train()