from cProfile import run
import copy
import csv
import sys
from cv2 import mean
from termcolor import colored
import numpy as np
import time
import pandas as pd


class Agent(object):

    def __init__(self, row, col, prob):
        self.position = [row, col]
        self.reward = 0
        self.prob = prob

    def takeAction(self, direction, walls, grid_row_size, grid_col_size):

        new_position = copy.copy(self.position)
        action = None
        if direction == "up":
            action = np.random.choice(['up', '2up', 'down'], p=[self.prob, (1-self.prob)/2, (1-self.prob)/2])
        if direction == "down":
            action = np.random.choice(['down', '2down', 'up'], p=[self.prob, (1-self.prob)/2, (1-self.prob)/2])
        if direction == "left":
            action = np.random.choice(['left', '2left', 'right'], p=[self.prob, (1-self.prob)/2, (1-self.prob)/2])
        if direction == "right":
            action = np.random.choice(['right', '2right', 'left'], p=[self.prob, (1-self.prob)/2, (1-self.prob)/2])

        if action == 'up':
            new_position[0] += -1
        elif action == 'down':
            new_position[0] += 1
        elif action == 'right':
            new_position[1] += 1
        elif action == 'left':
            new_position[1] += -1
        elif action == '2up':
            new_position[0] += -1
            barrier = new_position in walls or new_position[0] < 0 or new_position[0] >= grid_row_size or new_position[
                1] < 0 or new_position[1] >= grid_col_size
            if not barrier:
                self.position = copy.copy(new_position)
                new_position[0] += -1
        elif action == '2down':
            new_position[0] += 1
            barrier = new_position in walls or new_position[0] < 0 or new_position[0] >= grid_row_size or new_position[
                1] < 0 or new_position[1] >= grid_col_size
            if not barrier:
                self.position = copy.copy(new_position)
                new_position[0] += 1
        elif action == '2right':
            new_position[1] += 1
            barrier = new_position in walls or new_position[0] < 0 or new_position[0] >= grid_row_size or new_position[
                1] < 0 or new_position[1] >= grid_col_size
            if not barrier:
                self.position = copy.copy(new_position)
                new_position[1] += 1
        elif action == '2left':
            new_position[1] += -1
            barrier = new_position in walls or new_position[0] < 0 or new_position[0] >= grid_row_size or new_position[
                1] < 0 or new_position[1] >= grid_col_size
            if not barrier:
                self.position = copy.copy(new_position)
                new_position[1] += -1

        # checking for boundaries and walls
        barrier = new_position in walls or new_position[0] < 0 or new_position[0] >= grid_row_size or new_position[
            1] < 0 or new_position[1] >= grid_col_size

        if not barrier:
            self.position = new_position

        return self.position

    def epsilon_greedy(self, Q_grid, e):

        [y, x] = self.position
        actions = Q_grid[y][x]

        if np.random.uniform(0, 1) < e:
            return np.random.choice(list(actions.keys()))  # random direction
        else:
            return max(actions, key=actions.get)  # direction with maximum Q value


class Gridworld(object):

    def __init__(self, grid, alpha, gamma, default_reward, probability):

        self.init_grid = grid
        self.row = len(grid)
        self.col = len(grid[0])
        self.walls = []
        self.start_pos = None
        self.goal = None
        self.pit = None
        self.pit_penalty = None
        self.goal_reward = None
        self.default_reward = default_reward
        self.probability = probability
        self.alpha = alpha
        self.gamma = gamma

        for i in range(self.row):
            for j in range(self.col):
                if grid[i][j] != 0:
                    if grid[i][j] == 'S':
                        self.start_pos = [i, j]
                    elif grid[i][j] == 'X':
                        self.walls.append([i, j])
                    else:
                        if int(grid[i][j]) < 0:
                            self.pit = [i, j]
                            self.pit_penalty = int(grid[i][j])
                        elif int(grid[i][j]) > 0:
                            self.goal = [i, j]
                            self.goal_reward = int(grid[i][j])

        #  Q grid values to 0
        self.Q_grid = [[{'up': 0., 'right': 0., 'down': 0., 'left': 0.}
                        for _ in range(self.col)]
                       for _ in range(self.row)]

        self.frequency_grid = [[0
                        for _ in range(self.col)]
                       for _ in range(self.row)]

    def init_agent(self):
        self.agent = Agent(self.start_pos[0], self.start_pos[1], self.probability)

    def return_reward(self, position):
        # reward at a position
        if position == self.pit:
            return self.pit_penalty
        elif position == self.goal:
            return self.goal_reward
        else:
            return self.default_reward

    def run_episode(self, e):

        # initializing agent at start position
        self.init_agent()

        # Choose action from present state using epsilon-greedy policy
        direction = self.agent.epsilon_greedy(self.Q_grid, e=e)

        # Repeat until a terminal state is reached
        while self.agent.position != self.pit and self.agent.position != self.goal:
            [current_y, current_x] = self.agent.position
            self.frequency_grid[current_y][current_x] += 1

            self.agent.takeAction(direction, self.walls, self.row, self.col)
            reward = self.return_reward(self.agent.position)
            self.agent.reward += reward

            new_direction = self.agent.epsilon_greedy(self.Q_grid, e=0)     # For Q-learning, e=0 to choose the maximum direction for next state
            [new_y, new_x] = self.agent.position

            self.Q_grid[current_y][current_x][direction] += self.alpha * (
                    reward + self.gamma * self.Q_grid[new_y][new_x][new_direction] -
                    self.Q_grid[current_y][current_x][direction])

            direction = new_direction


def print_grid_world(grid):
    print("GRID WORLD:")
    for i in range(len(grid)):
        for j in range(len(grid[0])):
            print("%s\t" % grid[i][j], end="")
        print("")
    print("")


def final_policies(world):
    best_policies = []

    for y in range(world.row):
        best_policies.append([])
        for x in range(world.col):
            actions = world.Q_grid[y][x]
            best_direction = max(actions, key=actions.get)

            best_policies[y].append(best_direction)

    best_policies = np.matrix(best_policies)

    for wall in world.walls:
        best_policies[wall[0], wall[1]] = 'WALL'

    best_policies[world.pit[0], world.pit[1]] = 'PIT'
    best_policies[world.goal[0], world.goal[1]] = 'GOAL'

    return best_policies


def generate_heat_map(world):
    heat_map_freq = [[0
                for _ in range(world.col)]
                for _ in range(world.row)]

    total_frequency = 0
    for i in range(world.row):
        for j in range(world.col):
            total_frequency += world.frequency_grid[i][j]

    for i in range(world.row):
        for j in range(world.col):
            heat_map_freq[i][j] = (world.frequency_grid[i][j]/total_frequency)*100

    return heat_map_freq


def generate_graph(time_points, mean_rewards):
    df1 = pd.DataFrame.from_dict({"Time": time_points, "Mean Reward": mean_rewards})
    df1.to_csv('part2_data.csv', header=True, index=False)


def play(world, e, span_time):

    init_time = time.time()
    run_time = 0
    total_reward = 0
    episodes = 0
    time_points = []
    mean_rewards = []
    t_interval = -1

    while run_time < span_time:
        world.run_episode(e)
        episodes += 1
        total_reward += world.agent.reward
        run_time = time.time() - init_time
        
        if run_time - t_interval > 0.1:         # collecting the mean reward value for every 0.1 secs
            t_interval = run_time
            time_points.append(run_time)
            mean_rewards.append((total_reward/episodes))
        
    generate_graph(time_points, mean_rewards)
    final_policy = final_policies(world)
    final_heat_map = generate_heat_map(world)

    return final_policy, final_heat_map, total_reward, episodes


def print_heat_map(grid, heat_map):
    print('HEAT MAP:')
    for i in range(len(heat_map)):
        for j in range(len(heat_map[0])):
            if heat_map[i][j] > 0:
                print("%i\t" % round(heat_map[i][j]), end="")
            else:
                if grid[i][j] == 'X':
                    print(colored(grid[i][j], "blue") + '\t', end='')
                elif grid[i][j] == 's':
                    continue
                else:
                    if int(grid[i][j]) < 0:
                        print(colored(grid[i][j], "red") + '\t', end='')
                    elif int(grid[i][j]) > 0:
                        print(colored(grid[i][j], "green") + '\t', end='')
        print("")
    print("")


def print_grid_policy(world, grid_policy):
    print('GRID POLICY:')
    for i in range(grid_policy.shape[0]):
        for j in range(grid_policy.shape[1]):
            if grid_policy[i, j] == 'GOAL':
                a = str(world.goal_reward)
                print(colored(a, "green") + '\t', end='')
            if grid_policy[i, j] == 'left':
                a = '<'
                print("%s\t" % a, end="")
            if grid_policy[i, j] == 'right':
                a = '>'
                print("%s\t" % a, end="")
            if grid_policy[i, j] == 'up':
                a = '^'
                print("%s\t" % a, end="")
            if grid_policy[i, j] == 'down':
                a = 'v'
                print("%s\t" % a, end="")
            if grid_policy[i, j] == 'WALL':
                a = 'X'
                print(colored(a, "blue") + '\t', end='')
            if grid_policy[i, j] == 'PIT':
                a = str(world.pit_penalty)
                print(colored(a, "red") + '\t', end='')
        print("")
    print("")


def calculate_mean_reward(total_reward, trails):

    mean_reward = total_reward/trails
    print("Mean reward per trail = %.3f" % mean_reward)


def main():
    if len(sys.argv) != 6:
        print("Required format: part2.py <filename> <reward> <gamma> <time to learn> <movement probability>")
        exit(1)
    else:
        file = sys.argv[1]
        reward_per_action = float(sys.argv[2])
        gamma = float(sys.argv[3])
        time_to_learn = float(sys.argv[4])
        prob_of_moving = float(sys.argv[5])
    
    print("This program will read in", file)
    print("It will run for", time_to_learn, "seconds")
    print("Its decay rate is", gamma, "and the reward per action is", reward_per_action)
    print("Its transition model will move the agent properly with p =", prob_of_moving)

    alpha = 0.5  # learning rate for Q-learning
    epsilon = 0.3

    grid = list(csv.reader(open(file, "r"), delimiter='\t'))
    print_grid_world(grid)

    world = Gridworld(grid, alpha, gamma, reward_per_action, prob_of_moving)

    grid_policy, heat_map, total_reward, num_trails = play(world, epsilon, time_to_learn)
    print_grid_policy(world, grid_policy)
    print_heat_map(grid, heat_map)
    calculate_mean_reward(total_reward, num_trails)


main()