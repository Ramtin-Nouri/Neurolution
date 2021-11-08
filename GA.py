from re import A
from Config import *
from Agent import Agent
import time
import numpy as np
# No idead why but I need this to not get an error:
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)#
else:
    print("ATTENTION: No GPU was found!")


def create_agents():
    global agents
    agents = []
    for i in range(NUM_AGENTS):
        agents.append(Agent(ENV,i))

def episode():
    global agents
    for agent in agents:
        agent.run()

    fitness=[]
    for i in range(len(agents)):
        agents[i].thread.join()
        print(F"Fitness of agent {i}: {agents[i].fitness}")
        fitness.append(agents[i].fitness)
    return fitness
    
def choose_parents(fitness):
    """Returns indixes for parents of next generation.
    
    Arguments:
    ----------
    fitness: list
        list of fitness values of each agents

    Returns:
    --------
    partens: list of indexes
        (NUM_AGENTS - ELITE_SIZE) entries
    """
    number_of_parents = NUM_AGENTS - ELITE_SIZE
    fitness = np.array(fitness) + EPSILON # s.t. we don't divide by 0 if all values are 0
    probabilities = fitness/np.sum(fitness)
    idx = range(len(fitness)) # we only care about their indexes not their values
    parents = np.random.choice(idx , size=number_of_parents
        , p = probabilities)
    return parents

print("Creating Agents")
create_agents()

for i in range(EPISODES):
    print(F"Running Generation {i}")
    
    fitness = episode()
    
    # Declare Elite
    if ELITE_SIZE > 0:
        elite_idx = np.argsort(-np.array(fitness))[:ELITE_SIZE]
    else:
        elite_idx = []
    
    parent_idx = choose_parents(fitness)

    new_generation = []
    # Keep elite as is
    for idx in elite_idx:
        new_generation.append(agents[idx])

    for i in range(len(parent_idx)):
        child_agent = Agent(ENV,i)
        child_agent.copy_brain(agents[parent_idx[i]])
        child_agent.mutate()
        new_generation.append(child_agent)
    
    agents = new_generation
    
    time.sleep(3) #?
