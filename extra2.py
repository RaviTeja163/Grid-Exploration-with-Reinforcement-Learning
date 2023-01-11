import copy
import csv
from curses.ascii import isalpha
import sys
from termcolor import colored
import numpy as np
import time


class Agent(object):

    def __init__(self, row, col, prob):
        self.position = [row, col]
        self.reward = 0
        self.prob = prob

    def takeAction(self, direction, walls, wormholes, grid_row_size, grid_col_size, grid):
        new_position = copy.copy(self.position)

        if direction == 'up':
            new_position[0] += -1
        elif direction == 'down':
            new_position[0] += 1
        elif direction == 'right':
            new_position[1] += 1
        elif direction == 'left':
            new_position[1] += -1

        # checking for boundaries and walls
        barrier = new_position in walls or new_position[0] < 0 or new_position[0] >= grid_row_size or new_position[1] < 0 or new_position[1] >= grid_col_size
        if not barrier:
            self.position = new_position

        # checking for wormholes
        teleport = new_position in wormholes
        if teleport:
            letter = grid[new_position[0]][new_position[1]]
            for wormhole in wormholes:
                if grid[wormhole[0]][wormhole[1]] == letter:
                    if wormhole[0] == new_position[0] and wormhole[1] == new_position[1]:
                        continue
                    else:
                        self.position = wormhole
                        break

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
        self.wormholes = []
        self.start_pos = None
        self.goal = []
        self.pit = []
        self.pit_penalty = []
        self.goal_reward = []
        self.default_reward = default_reward
        self.probability = probability
        self.alpha = alpha
        self.gamma = gamma

        for i in range(self.row):
            for j in range(self.col):
                if grid[i][j] != 0:
                    if grid[i][j] == 'S':               # starting position
                        self.start_pos = [i, j]
                    elif grid[i][j] == 'X':             # walls
                        self.walls.append([i, j])
                    elif grid[i][j].isalpha():          # any other letter will be considered as wormholes
                        self.wormholes.append([i, j])
                    else:
                        if int(grid[i][j]) < 0:         # pitfall
                            self.pit.append([i, j])
                            self.pit_penalty.append(int(grid[i][j]))
                        elif int(grid[i][j]) > 0:       # treasure
                            self.goal.append([i, j])
                            self.goal_reward.append(int(grid[i][j]))

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
        if position in self.pit:
            return int(self.init_grid[position[0]][position[1]])
        elif position in self.goal:
            return int(self.init_grid[position[0]][position[1]])
        else:
            return self.default_reward

    def run_episode(self, e):

        # initializing agent at start position
        self.init_agent()

        # Repeat until a terminal state is reached
        while self.agent.position not in self.pit and self.agent.position not in self.goal:
            # Choose action from present state using epsilon-greedy policy
            direction = self.agent.epsilon_greedy(self.Q_grid, e=e)

            [current_y, current_x] = self.agent.position
            self.frequency_grid[current_y][current_x] += 1

            self.agent.takeAction(direction, self.walls, self.wormholes, self.row, self.col, self.init_grid)
            reward = self.return_reward(self.agent.position)
            self.agent.reward += reward

            new_direction = self.agent.epsilon_greedy(self.Q_grid, e=0)     # For Q-learning, e=0 to choose the maximum direction for next state
            if self.default_reward == 0:
                new_direction = self.agent.epsilon_greedy(self.Q_grid, e=e)

            [new_y, new_x] = self.agent.position

            self.Q_grid[current_y][current_x][direction] += self.alpha * (
                    reward + self.gamma * self.Q_grid[new_y][new_x][new_direction] -
                    self.Q_grid[current_y][current_x][direction])


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

    if len(world.pit) > 0:
        for i in range(len(world.pit)):
            best_policies[world.pit[i][0], world.pit[i][1]] = 'PIT'
    for i in range(len(world.goal)):
        best_policies[world.goal[i][0], world.goal[i][1]] = 'GOAL'

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


def play(world, e, span_time):

    init_time = time.time()
    run_time = init_time
    rewards = []
    
    while run_time - init_time < span_time:
        world.run_episode(e)
        rewards.append(world.agent.reward)
        run_time = time.time()

    final_policy = final_policies(world)
    final_heat_map = generate_heat_map(world)

    return final_policy, final_heat_map, rewards


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
                    else:
                        print("%i\t" % round(heat_map[i][j]), end="")                        
        print("")
    print("")


def print_grid_policy(world, grid_policy):
    print('GRID POLICY:')
    for i in range(grid_policy.shape[0]):
        for j in range(grid_policy.shape[1]):
            if grid_policy[i, j] == 'GOAL':
                a = str(world.init_grid[i][j])
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
                a = str(world.init_grid[i][j])
                print(colored(a, "red") + '\t', end='')
        print("")
    print("")


def calculate_mean_reward(rewards):
    trails = len(rewards)
    total_reward = 0
    for reward in rewards:
        total_reward += reward

    average_reward = total_reward/trails
    print("Mean reward per trail = %.3f" % average_reward)


def main():
    if len(sys.argv) != 6:
        print("Required format: extra2.py <filename> <reward> <gamma> <time to learn> <movement probability>")
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
    epsilon = 0.2

    grid = list(csv.reader(open(file, "r"), delimiter='\t'))
    print_grid_world(grid)

    world = Gridworld(grid, alpha, gamma, reward_per_action, prob_of_moving)

    grid_policy, heat_map, all_rewards = play(world, epsilon, time_to_learn)
    print_grid_policy(world, grid_policy)
    print_heat_map(grid, heat_map)
    calculate_mean_reward(all_rewards)


main()