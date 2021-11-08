import numpy as np
from Model import Brain
from pygad import kerasga
class DNA():
    """
        Representation of genetic code.

        Retrieves weights from model and consideres them the DNA.
        
        Attributes:
        -----------
        mutation_factor: float
            probability for each dna value to randomly change
        brain: Brain
            holds Keras model
        vector: list
            list representation of the weights

    """
    def __init__(self,brain):
        self.mutation_factor = 0.05
        self.initializeRandom()
        self.brain = brain

    def initializeRandom(self):
        pass

    def get_dna(self):
        self.vector = kerasga.model_weights_as_vector(self.brain.model)

    def crossover(self, otherDNA):
        pass

    def mutate(self):
        mask = np.random.choice([0, 1],
         size=len(self.vector), p=((1 - self.mutation_factor), self.mutation_factor)).astype(np.bool)
        rands = np.random.uniform(low=-1,high=1,size=len(self.vector))
        self.vector[mask] = rands[mask]
