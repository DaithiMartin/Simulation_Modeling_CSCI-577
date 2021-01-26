import numpy as np
from time import sleep
from IPython.display import clear_output
from matplotlib import pyplot as plt
# %matplotlib inline



class DLA:

    def __init__(self, world_area=61):
        self.L = world_area // 2
        self.lattice = self.lattice = np.zeros((world_area, world_area), dtype=np.int8).tolist()
        self.lattice[self.L][self.L] = 1

        self.num_particles = 1
        self.ring_size = self.L // 10
        self.start_radius = 3
        self.max_radius = self.ring_size + self.start_radius

        self.cycles = 0
        self.complete = False

    def reset(self):
        self.__init__()
        return None

    def iterate(self):
        x = 0
        y = 0
        if self.start_radius < self.L:

            theta = 2 * np.pi * np.random.random()
            x = self.L + int(self.start_radius * np.cos(theta))
            y = self.L + int(self.start_radius * np.sin(theta))
            self.walk(x, y)

            return None

        else:
            self.complete = True

        self.cyclyes += 1

        return None

    def walk(self, x, y):

        walking = True

        while walking:
            r_squared = (x - self.L) ** 2 + (y - self.L) ** 2
            r = 1 + int(np.sqrt(r_squared))
            if r > self.max_radius:
                walking = False
            try:
                if r < self.L and (self.lattice[x+1][y] + self.lattice[x-1][y] + self.lattice[x][y+1] + self.lattice[x][y-1] > 0):
                    self.num_particles += 1
                    self.lattice[x][y] = 1
                    if r >= self.start_radius:
                        self.start_radius = r+2
                    self.max_radius = self.start_radius + self.ring_size
                    walking = False
            except IndexError:
                pass

            else:
                direction = np.random.randint(0,4)
                if direction == 0:
                    x += 1
                elif direction == 1:
                    x -= 1
                elif direction == 2:
                    y += 1
                else:
                    y -= 1

        return None

    def animate(self):
        """
        Iterate through the space until cluster is completely surrounded by dead cells.

        :return: None
        """
        plt.figure()
        while not self.complete:
            self.plot()
            self.iterate()
            sleep(0.01)
            clear_output(wait=True)
        self.plot()

        return None

    def plot(self):
        """
        Plot current lattice configuration.

        :return: None
        """
        plt.figure()
        plt.title("World Size: {}x{}\nIteration: {}".format(self.L,
                                                            self.L,
                                                            self.cycles))

        plt.imshow(self.lattice)
        plt.show()

        return None

#%%
DLA = DLA()

for i in range(100):
    DLA.iterate()


# DLA.iterate()
DLA.plot()


