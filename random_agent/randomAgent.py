import numpy as np
import random
import matplotlib.pyplot as plt
from minesweeper import core

class RandomMinesweeperAgent:
    def __init__(self, env, num_episodes=1000):
        """
        Initializes the Random Agent for Minesweeper.

        Args:
            env (Board): Instance of Minesweeper board.
            num_episodes (int): Number of games to play.
        """
        self.env = env
        self.num_episodes = num_episodes
        self.rewards_per_ep = []
        self.moves_per_ep = []
        self.win_count = 0

    def get_state(self):
        """Returns a simplified state representation."""
        return tuple(
            tuple(self.env._tiles[i][j].type for j in range(self.env.cols)) 
            for i in range(self.env.rows)
        )

    def avail_actions(self):
        """Returns all unopened tiles as available actions."""
        return [
            (i, j) for i in range(self.env.rows) 
            for j in range(self.env.cols) 
            if self.env.tile_valid(i, j) and self.env._tiles[i][j] == core.BoardTile.unopened
        ]

    def random_action(self):
        """Selects a random available action (tile to reveal)."""
        actions = self.avail_actions()
        return random.choice(actions) if actions else None

    def reward_system(self, i, j):
        """Defines a simple reward system."""
        reward = 0
        opened_tiles = self.env.tile_open(i, j)

        if not opened_tiles:
            return reward

        if self.env.is_game_over:
            reward = -20  # Huge penalty for hitting a mine
        elif self.env.is_game_finished:
            reward = 100  # Large reward for solving the board
            self.win_count += 1
        else:
            for tile in opened_tiles:
                if tile.type != core.BoardTile.mine:
                    reward += 1  # Reward for revealing safe tiles

        return reward

    def play(self):
        """Runs the random agent through multiple games."""
        for ep in range(self.num_episodes):
            self.env.game_new(self.env.rows, self.env.cols, self.env.mines)
            done = False
            ep_reward = 0
            move_count = 0

            while not done:
                action = self.random_action()
                if action is None:
                    break
                
                i, j = action
                reward = self.reward_system(i, j)
                ep_reward += reward
                move_count += 1

                if self.env.is_game_over or self.env.is_game_finished:
                    done = True

            self.rewards_per_ep.append(ep_reward)
            self.moves_per_ep.append(move_count)

    def plot_rewards(self):
        """Plots the total rewards over episodes."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.rewards_per_ep)
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.title("Random Agent: Rewards Over Time")
        plt.show()

    def plot_moves(self):
        """Plots the number of moves taken per episode."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.moves_per_ep)
        plt.xlabel("Episodes")
        plt.ylabel("Number of Moves")
        plt.title("Random Agent: Moves Per Episode")
        plt.show()

    def print_summary(self):
        """Prints the final results after all episodes."""
        win_rate = (self.win_count / self.num_episodes) * 100
        avg_moves = sum(self.moves_per_ep) / self.num_episodes
        avg_reward = sum(self.rewards_per_ep) / self.num_episodes

        print("RANDOM AGENT RESULTS:")
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Average Moves Per Game: {avg_moves:.2f}")
        print(f"Average Reward: {avg_reward:.2f}")



