"""
Collection of all global variables
"""

# Number of individuals
NUM_AGENTS = 25
# Number of total generations
EPISODES = 1000
# OpenAi Gym Environment
ENV = "Pong-v4"
# Use Custom Environment Wrapper
WRAP_ENV = True
# Number of individuals to keep unchanged
ELITE_SIZE = 2
# Number of completely new individuals introduced
ALIENS = 2
# Number of maximal simulation steps after which to stop
MAX_SIM_STEPS = 500
# Constant to add whenever we divide where the variable could be 0
EPSILON = 1e-8
# Mutation Factor: Probabilty per value to randomly change
MUTATION_FACTOR = 0.2
