import numpy as np
import random
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm
from minesweeper import core


class TDLearning:
    def __init__(self, env, alpha, gamma, epsilon, num_eps):
        """
        Initialize MineSweeper Board environment.
        
        Arg:
            env (Board): Instance of Board environment.
        
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {} # Q-values
        self.rewards_per_ep = []
        self.num_eps = num_eps
        self.ep_steps = []
        self.num_wins = 0
        

    def get_state(self):
        """
        Convert Board into tuple representation
        """
        state = str(self.env)
        return state

    def avail_actions(self, board):
        """
        Unopened BoardTile will represent actions in the form of a tuple
        """
        actions = []
        
        for i in range(board.rows):
            for j in range(board.cols):
                if board.tile_valid(i,j) and (board._tiles[i][j] == core.BoardTile.unopened):
                    actions.append((i,j))
        return actions
        
    def apply_action(self, state, board):
        """
        Choose action based on epsilon-greedy policy
        """
        actions = self.avail_actions(board)
        
        if not actions:
            return None

        #additional heuristic to start with corners first
        corners = [(0,0), (0, board.cols -1), (board.rows-1, 0), (board.rows-1, board.cols -1)]
        avail_corners = [a for a in actions if a in corners]
        
        if avail_corners:
                action = random.choice(avail_corners)
                return action
        
        if np.random.rand() < self.epsilon:
            action = random.choice(actions)
            return action

        action = max(self.q_table[state], key = self.q_table[state].get)
        return action

    def update_q_val(self, state, action, reward, next_state):
        if state not in self.q_table:
            self.q_table[state] = {}

        if action not in self.q_table[state]:
            self.q_table[state][action] = 0

        max_future_q = max(self.q_table.get(next_state, {}).values(), default = 0)
        
        self.q_table[state][action] += self.alpha * (reward + self.gamma * max_future_q - self.q_table[state][action])
        
    def reward_sys(self, board, i, j):
        """
        Reward system
        """
        reward = 0
        
        opened_tiles = board.tile_open(i,j)
        corners = [(0,0), (0, board.cols -1), (board.rows-1, 0), (board.rows-1, board.cols -1)]

        if not opened_tiles:
            return reward

        if board.is_game_over:
            reward = -20
            
        elif board.is_game_finished:
            self.num_wins += 1
            reward = 100
            
        else:
            reward += len(opened_tiles)

            #Bonus: reward for opening corners
            if (i,j) in corners:
                reward += 15

            #Bonus: Reward for opening a tile **near a tile with a "1" bomb
            if self._is_near_one(i,j):
                reward += 10

        return reward

    def _is_near_one(self, row, col):
        """
        Checks if the given tile (row, col) is adjacent to a tile with a '1'.
        """
        for i in [-1, 0, 1]:  # Check adjacent rows
            for j in [-1, 0, 1]:  # Check adjacent columns
                if i == 0 and j == 0:
                    continue  # Skip the tile itself

                new_row, new_col = row + i, col + j
                if 0 <= new_row < self.env._rows and 0 <= new_col < self.env._cols:
                    if self.env._tiles[new_row][new_col].type == "1":
                        return True  # Found a '1' nearby

        return False  # No '1' nearby
        
    def train(self):
        for ep in tqdm(range(self.num_eps), desc="Training Progress"):
            curr_board = copy.deepcopy(self.env)
            state = self.get_state()
            done = False
            ep_reward = 0
            step = 0
            
            while not done:
                action = self.apply_action(state, curr_board)
                step += 1
                
                if action is None:
                    break
                    
                i, j = action
                
                next_state = self.get_state()

                r = self.reward_sys(curr_board,i,j)

                ep_reward += r

                self.update_q_val(state, action, r, next_state)

                
                if curr_board.is_game_over or curr_board.is_game_finished:
                    done = True
                    
                state = next_state

            self.rewards_per_ep.append(ep_reward)
            self.ep_steps.append(step)

    def plot_rewards(self):
        plt.figure(figsize=(15, 8))
        plt.plot(self.rewards_per_ep)
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.title("TD Learning Rewards Over Time")
        plt.show()

        
        
        plt.figure(figsize=(15, 8))
        plt.plot(self.ep_steps)
        plt.xlabel("Episodes")
        plt.ylabel("Steps")
        plt.title("TD Learning: Steps per Episode")
        plt.show()

        print("TD Learning Agent")
        print("Win rate: ", self.num_wins/self.num_eps)
        print("Avg Steps: ", sum(self.ep_steps)/self.num_eps)
        print("Avg rewards: ", sum(self.rewards_per_ep)/self.num_eps)


        


        