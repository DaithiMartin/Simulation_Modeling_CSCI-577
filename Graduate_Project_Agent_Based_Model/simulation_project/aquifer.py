
class Aquifer:

    def __init__(self, initial_volume):
        self.available_volume = initial_volume
        self.recharge_mu = initial_volume * 0.1
        self.recharge_sigma = self.recharge_mu * 0.3

    def withdraw_water(self, amount):
        """
        Withdraws water from the aquifer.
        Currently allows for complete draining of the aquifer but prevents negative values.

        Amount in the aquifer is unknown to the farmer and therefore is not in the agent's state vector.
        :param amount: attempted water withdrawal
        :return: actual amount withdrawn
        """
        if amount < self.available_volume:
            self.available_volume -= amount

            return amount

        else:
            amount = self.available_volume
            self.available_volume = 0

            return amount

    def recharge_aquifer(self):
        """
        Stochastic recharge of the aquifer.
        Currently based in independent distribution but could be tied to surface water.
        :return: None
        """
        self.available_volume += np.random.randn() * self.recharge_sigma + self.recharge_mu

        return None
