import numpy as np
from Model import Brain
from pygad import kerasga
class DNA():

    #code = []
    def __init__(self,brain):
        self.mutationFactor = 0.05
        self.initializeRandom()
        self.brain = brain

    def initializeRandom(self):
        pass

    def get_dna(self,weights):
        self.vector = kerasga.model_weights_as_vector(self.brain.model)

    def crossover(self, otherDNA):
        pass

    def mutate(self):
        mask = np.random.choice([0, 1], size=len(self.vector), p=((1 - self.mutationFactor), self.mutationFactor)).astype(np.bool)
        rands = np.random.uniform(low=-1,high=1,size=len(self.vector))
        self.vector[mask] = rands[mask]


