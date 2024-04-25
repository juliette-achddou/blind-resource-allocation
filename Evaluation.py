import numpy as np

from Result import Result


class Evaluation:
    """The Evaluation class for analyzing the output of several Monte Carlo experiments."""

    def __init__(self, env, pol, nbRepetitions, horizon, tsav=[], dim=1, print_=False):
        if len(tsav) > 0:
            self.tsav = tsav
        else:
            self.tsav = np.arange(horizon)
        self.env = env
        self.dim = env.constraints.dim
        self.pol = pol
        self.nbRepetitions = nbRepetitions
        self.horizon = horizon
        self.cumReward = np.zeros((self.nbRepetitions, len(tsav)))
        self.cumBestReward = np.zeros((self.nbRepetitions, len(tsav)))
        self.cumRegret = np.zeros((self.nbRepetitions, len(tsav)))
        self.distance_to_opti = np.zeros((self.nbRepetitions, len(tsav)))
        self.choices = np.zeros((self.nbRepetitions, len(tsav), self.dim))
        self.it_choices = np.zeros((self.nbRepetitions, len(tsav), self.dim))
        self.resultUnique = Result(horizon, dim)
        self.print_ = print_
        for k in range(nbRepetitions):
            if print_ is True:
                if nbRepetitions < 10 or k % int((nbRepetitions / 10)) == 0:
                    print(k)
            result = env.play(self.pol, self.horizon)
            if k == 1:
                self.resultUnique = result
            self.cumReward[k, :] = np.cumsum(result.rewards)[tsav]
            self.cumRegret[k, :] = np.cumsum(result.regret)[tsav]
            self.choices[k, :, :] = result.choices[tsav]
            self.distance_to_opti[k, :] = result.distance_to_opti[tsav]
            if self.pol.iterate_bool:
                self.it_choices[k, :, :] = result.it_choices[tsav]

    def meanReward(self):
        return sum(self.cumReward[:, -1]) / len(self.cumReward[:, -1])

    def meanDistance(self):
        return np.mean(self.distance_to_opti, 0)

    def meanRegret(self):
        return np.mean(self.cumRegret, 0)
