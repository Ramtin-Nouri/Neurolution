import numpy as np
import random
from Config import MUTATION_FACTOR_SUBSTITUTE, MUTATION_FACTOR_DUPLICATE, MUTATION_FACTOR_INSERT_DELETE, MUTATION_FACTOR_INVERT
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
        self.initializeRandom()
        self.brain = brain
        self.vector = []

    def initializeRandom(self):
        pass

    def get_dna(self):
        self.vector = kerasga.model_weights_as_vector(self.brain.model)

    def set_dna(self):
        kerasga.model_weights_as_matrix(self.brain.model, self.vector)

    def crossover(self, otherDNA):
        self.get_dna()
        otherDNA.get_dna()

        mask = np.random.choice([0, 1],
         size=len(self.vector), p=((0.5 , 0.5)).astype(np.bool))
        self.vector[mask] = otherDNA.vector[mask]

        self.set_dna()
        otherDNA.set_dna()

    def mutate(self):
        """Mutation is implemented inspired by different kinds of bio-inspired types.

        The weights of the neural network are converted to a vector
        and can therefor be handled analogous the the DNA sequence.
        
        I implement these types of mutation:
        1. Substitution: replaces values randomly with new ones.
        2. Insertion: randomly inserts a new value into the sequence
        3. Deletion: randomly delete a value from the sequence
        4. Duplication: one or multiple values or copied and repeated. Replacing the following values.
        5. Inversion: a sequence of values is inverted in their order
        """
        self.get_dna()
        self.mutation_substitute()
        if random.random()< MUTATION_FACTOR_INSERT_DELETE:
            self.mutation_insert_delete()
        if random.random()< MUTATION_FACTOR_DUPLICATE:
            self.mutation_duplicate()
        if random.random()< MUTATION_FACTOR_INVERT:
            self.mutation_invert()
        self.set_dna()

    def mutation_substitute(self):
        mask = np.random.choice([0, 1],
         size=len(self.vector), p=((1 - MUTATION_FACTOR_SUBSTITUTE), MUTATION_FACTOR_SUBSTITUTE)).astype(np.bool)
        rands = np.random.uniform(low=-1,high=1,size=len(self.vector))
        self.vector[mask] = rands[mask]

    def mutation_insert_delete(self):
        ind_delete, ind_insert = random.sample(range(len(self.vector)), 2)
        self.vector = np.insert(self.vector, ind_insert, random.uniform(-1,1))
        self.vector = np.delete(self.vector,ind_delete)

    def mutation_duplicate(self):
        ind_dup = random.choice(range(len(self.vector)))
        rep = random.randint(1,3)
        l = len(self.vector)
        seq_len = max(1, min((l-ind_dup)//(rep+1), int(random.normalvariate(5,3)), l//10))

        #print(ind_dup,rep,seq_len)
        # duplicate
        for r in range(rep):

            #print(self.vector[ind_dup+(r+1)*seq_len:ind_dup+(r+2)*seq_len], self.vector[ind_dup:ind_dup+seq_len])
            self.vector[ind_dup+(r+1)*seq_len:ind_dup+(r+2)*seq_len] = self.vector[ind_dup:ind_dup+seq_len]

    def mutation_invert(self):
        ind_inv = random.choice(range(len(self.vector)))
        l = len(self.vector)
        seq_len = max(2, min((l-ind_inv), int(random.normalvariate(5,3)), l//10))
        self.vector[ind_inv:ind_inv+seq_len] = self.vector[ind_inv:ind_inv+seq_len][::-1]