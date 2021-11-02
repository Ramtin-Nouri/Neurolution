import numpy as np

class DNA():

    #code = []
    def __init__(self):
        self.mutationFactor = 0.05
        self.initializeRandom()

    def initializeRandom(self):
        pass #done anyway?

    def setGens(self,weights):
        self.gens = weights

    def crossover(self, otherDNA):
        pass

    def mutate(self):
        for gen in range(self.gens):
            if len(gen)<2:continue
            weights = gen[0]
            biases = gen[1]

            mask = np.random.choice([0, 1], size=weights.shape, p=((1 - self.mutationFactor), self.mutationFactor)).astype(np.bool)
            r = np.random.rand(*weights.shape)*np.max(weights)
            weights[mask] = r[mask]

            mask = np.random.choice([0, 1], size=biases.shape, p=((1 - self.mutationFactor), self.mutationFactor)).astype(np.bool)
            r = np.random.rand(*biases.shape)*np.max(biases)
            biases[mask] = r[mask]


