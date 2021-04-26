import numpy as np
import matplotlib.pyplot as plt
from simulation_project.prioritized_dqn_agent import Agent


# %%


class SimConfig:
    """
    This config class simplifies the initialization of the SimulationCES class.
    """

    def __init__(self):
        # general simulation parameters
        self.num_years = 2000

        # crop parameters
        self.num_crops = 1
        self.crop_1_price = 10
        self.crop_1 = {"pi": 10,
                       "beta_land": 0.5,
                       "beta_water": 0.5,
                       "rho": 1,
                       "r": 1}

        # hydrology function, current estimate based on lower Clark Fork
        self.water_mu = 0.75e6  # cfs
        self.water_sigma = 1e2  # cfs

        # stochastic water dist
        self.water_dist = self.water_sigma * np.random.randn(100) + self.water_mu
        # # constant water dist
        # self.water_dist = np.full(100, self.water_mu)

        # simulation parameters
        self.number_farmers = 3
        self.farmer_priority = [0, 1, 2]
        # self.random_seed = 1    # seems like decent seed
        self.random_seed = None

        # agent parameters
        self.state_size = 4
        self.action_size = 2 * self.num_crops
        self.memory_size = 10

        # land and water physical constraints
        # land available to each farmer
        self.available_land = [100 for _ in range(self.number_farmers)]

        # max water for each farmer, physical equipment constraint
        self.max_yearly_water = [1.5e6 for _ in range(self.number_farmers)]

        # experimental parameters
        self.epsilon_test = False

#%%


class SimulationCES:
    """
    Simulation environment utilizing CES production funtion
    """

    def __init__(self, config: SimConfig):
        # simulation configuration
        self.config = config

        # simulation records for analysis
        self.farmers_rewards_record = [[] for _ in range(config.number_farmers)]
        self.water_record = {"Surface": []}
        self.farmers_water_withdrawal_record = [[] for _ in range(config.number_farmers)]
        self.farmers_actions_record = [[] for _ in range(config.number_farmers)]
        self.farmer_available_water_record = [[] for _ in range(config.number_farmers)]

        # simulation parameters
        self.num_farmers = config.number_farmers
        self.farmer_priority = config.farmer_priority
        self.farmer_list = [Agent(config.state_size, config.action_size, config.random_seed) for _ in
                            range(self.num_farmers)]

        # simulation state parameters
        self.year = 1
        self.num_years = config.num_years

        # simulation functions, reward function is a property
        self.hydrology_function = config.water_dist

        # initialize river continuum
        self.source_water = self.init_available_water()
        self.available_water = [self.source_water for _ in range(self.num_farmers)]

        # land available to each farmer
        self.available_land = config.available_land

        # max water for each farmer, physical equipment constraint
        self.max_yearly_water = config.max_yearly_water

        # production function parameters
        self.crop_1 = config.crop_1
        self.crop_1_price = config.crop_1_price

        # experimental parameters
        self.epsilon_test = config.epsilon_test

    def reset(self):
        """
        Resets the simulation environment. Open AI Gym interface.

        Only the river continuum needs to be reset each episode.
        Everything else needs to persist between episodes. I think...

        Returns: None
        """

        # re-initialize available water, this includes surface
        self.source_water = self.init_available_water()
        self.available_water = [self.source_water for _ in range(self.num_farmers)]

        return None

    def reward_function(self, action):

        x1_water, x1_land = action

        """input amounts are the action space for the agents"""
        q_1 = self.crop_1["pi"] * (
                self.crop_1["beta_land"] * (x1_land ** self.crop_1["rho"]) + self.crop_1["beta_water"] * (
                x1_water ** self.crop_1["rho"])) ** (self.crop_1["r"] / self.crop_1["rho"])

        # calc rewards
        r_1 = np.log(q_1 * self.crop_1_price + 1e-3)

        land_cost = self.land_cost_function(x1_land)
        water_cost = self.water_cost_function(x1_water)
        total_cost = land_cost + water_cost

        shadow_price = 0

        return r_1 - total_cost - shadow_price

    def water_cost_function(self, total_water):
        """
        Linear cost for water.
        :param total_water: total water used for the farmer.
        :return: total water cost
        """
        w_0 = 3e-6
        cost = w_0 * total_water

        return cost

    def land_cost_function(self, total_land):
        """
        Linear Cost for Land
        :param total_land: total land used by the farmer.
        :return: total land cost
        """
        w_0 = 3e-6

        cost = w_0 * total_land

        return cost

    def plot_reward(self, crop):

        x_len = int(2e6)
        water = np.arange(x_len) + 1
        land = np.full(x_len, 100)

        # combine water and land
        action_vec = np.stack((water, land), axis=-1)

        y = []
        for action in action_vec:
            y.append(self.reward_function(action))

        x = action_vec[:, 0]

        plt.title("Reward Function")
        plt.xlabel("Water")
        plt.ylabel("Reward")
        plt.plot(x, y)
        plt.show()

        return None

    def init_available_water(self):
        """
        Initializes available water for beginning of episode.
        Pulls random sample from historic discharge distribution.

        This needs to be a function call so that we get a new random number each time we want to initialize
        a seasons available water.

        Returns: (float) available water.
        """
        # get initial values for water types
        surface_water = np.random.choice(self.hydrology_function)

        # update water records
        self.water_record["Surface"].append(surface_water)

        return surface_water

    def update_available_water(self, priority_index, action):
        """
        Updates all available water down stream from agent indicated by priority index.

        Args:
            priority_index: indicates at what point in the river continuum to update flows
            action: amount of water removed by agent
        """
        # determine the total water removed
        total_removed = action[0]
        self.farmers_water_withdrawal_record[priority_index].append(total_removed)

        # determine how much water needs to come from surface water
        surface_removed = total_removed

        # remove water from surface continuum, max operation prevents negative water amounts
        for i in range(priority_index, len(self.available_water)):
            self.available_water[i] = max(self.available_water[i] - surface_removed, 1e-3)

        return None

    def step(self):
        """
        Takes a step in the environment and updates all relevant instance attributes.

        This will look a little different than most other RL training loops for two reasons:
        1. This is inherently a multi agent system
        2. There is only one step in each episode

        Returns: None
        """

        # iterate through farmers in their priority order
        for priority_num in self.farmer_priority:
            agent = self.farmer_list[priority_num]

            # epsilon test
            if self.epsilon_test and 1100 > self.year > 1000:
                agent.eps = 0.75

            state = np.concatenate((
                [self.available_water[priority_num]],
                [min(self.max_yearly_water[priority_num], self.available_water[priority_num])],
                [self.available_land[priority_num]],
                [self.crop_1_price])
            )

            # record available water for this farmer
            self.farmer_available_water_record[priority_num].append(self.available_water[priority_num])

            # get action, and record actions for debugging
            action = agent.act(state)
            self.farmers_actions_record[priority_num].append(action)

            reward = self.reward_function(action)
            self.update_available_water(priority_num, action)

            # save reward for later analysis
            self.farmers_rewards_record[priority_num].append(reward)

            # estimate next action and save state information in agent memory
            next_state = np.concatenate((
                [self.available_water[priority_num]],
                [min(self.max_yearly_water[priority_num], self.available_water[priority_num])],
                [self.available_land[priority_num]],
                [self.crop_1_price])
            )
            agent.step(state, action, reward, next_state, True)

        # 1 year of simulation complete, reset the source water
        self.reset()
        self.year += 1


# for debugging
config = SimConfig()
config.farmer_priority = [0, 1, 2]
env = SimulationCES(config)

for i_episode in range(1, env.num_years + 1):
    env.step()

# red and green should be screwed

plt.xlabel("Iteration")
plt.ylabel("Reward")

average_rewards = []
for i, record in enumerate(env.farmers_rewards_record):
    average_rewards.append(np.convolve(record, np.ones(100), 'valid') / 100)

plt.plot(np.arange(len(average_rewards[0])), average_rewards[0], 'b-', label="Farmer 1")
plt.plot(np.arange(len(average_rewards[1])), average_rewards[1], 'r-', label="Farmer 2")
plt.plot(np.arange(len(average_rewards[2])), average_rewards[2], 'g-', label="Farmer 3")

plt.legend()
plt.show()

plt.xlabel("Iteration")
plt.ylabel("Total Water Withdrawn")

average_water_withdrawal = []
for i, record in enumerate(env.farmers_water_withdrawal_record):
    average_water_withdrawal.append(np.convolve(record, np.ones(100), 'valid') / 100)

plt.plot(np.arange(len(average_water_withdrawal[0])), average_water_withdrawal[0], 'b-', label="Farmer 1")
plt.plot(np.arange(len(average_water_withdrawal[1])), average_water_withdrawal[1], 'r-', label="Farmer 2")
plt.plot(np.arange(len(average_water_withdrawal[2])), average_water_withdrawal[2], 'g-', label="Farmer 3")
plt.legend()
plt.show()

# %%
plt.xlabel("Iteration")
plt.ylabel("Water Available to for each Farmer")

average_available_water = []
for i, record in enumerate(env.farmer_available_water_record):
    average_available_water.append(np.convolve(record, np.ones(100), 'valid') / 100)

plt.plot(np.arange(len(average_available_water[0])), average_available_water[0], 'b-', label="Farmer 1")
plt.plot(np.arange(len(average_available_water[1])), average_available_water[1], 'rx-', label="Farmer 2")
plt.plot(np.arange(len(average_available_water[2])), average_available_water[2], 'g-', label="Farmer 3")
plt.legend()
plt.show()

env.plot_reward(crop="crop 1")
