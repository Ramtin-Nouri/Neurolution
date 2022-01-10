"""
Collection of all global variables
"""

# Number of individuals
NUM_AGENTS = 200
# Number of total generations
EPISODES = 200
# OpenAi Gym Environment
ENV = "Pong-v4"
# Use Custom Environment Wrapper
WRAP_ENV = True
# Number of individuals to keep unchanged
ELITE_SIZE = 20
# Number of completely new individuals introduced
ALIENS = 10
# Number of maximal simulation steps after which to stop
MAX_SIM_STEPS = 2000
# Constant to add whenever we divide where the variable could be 0
EPSILON = 1e-8
# Mutation Factors:
MUTATION_FACTOR_SUBSTITUTE = 0.1
MUTATION_FACTOR_INSERT_DELETE = 0
MUTATION_FACTOR_DUPLICATE = 0
MUTATION_FACTOR_INVERT = 0
# Type of Model: choice between Dense and CNN
MODELTYPE = "Dense"
# Whether to have asexual reproduction or include crossover
IS_ASEXUAL = True
