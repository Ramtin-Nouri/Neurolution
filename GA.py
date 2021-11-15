from Config import *
from Agent import Agent
import time
import numpy as np
import os
from datetime import datetime
import signal
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
    number_of_parents = NUM_AGENTS - ELITE_SIZE - ALIENS
    fitness = np.array(fitness) + EPSILON # s.t. we don't divide by 0 if all values are 0
    if np.min(fitness) < 0 :
        # offset in case values are in the negatives
        fitness -= np.min(fitness) 
    probabilities = fitness/np.sum(fitness)
    idx = range(len(fitness)) # we only care about their indexes not their values
    parents = np.random.choice(idx , size=number_of_parents, p = probabilities)
    return parents

def init():
    # Create folder for logging
    today = datetime.today()
    time = "%04d-%02d-%02d-%02d-%02d-%02d/" % (today.year,today.month,today.day,today.hour,today.minute,today.second)
    os.makedirs(F"savedata/{time}")
    file_writer = tf.summary.create_file_writer(F"savedata/{time}")
    file_writer.set_as_default()
    # Log some configs
    tf.summary.text("Environment", ENV, step=0)
    tf.summary.text("Number of Agents", str(NUM_AGENTS), step=0)
    tf.summary.text("Number of Elites", str(ELITE_SIZE), step=0)
    tf.summary.text("Number of New Agents", str(ALIENS), step=0)
    tf.summary.text("Maximum Simulation Steps", str(MAX_SIM_STEPS), step=0)
    tf.summary.text("Mutation Factor", str(MUTATION_FACTOR), step=0)
    # Create Agents
    print("Creating Agents")
    create_agents()

def close_signal(sig,frame):
    global should_run
    print("Ctrl-C detected. Quitting...")
    should_run = False
    for agent in agents:
        agent.should_run = False
    for agent in agents:
        agent.thread.join()

def log_stats(fitnessList,step):
    tf.summary.scalar("Best Scoring Individual", np.max(fitnessList),step=step)
    tf.summary.scalar("Worst Scoring Individual", np.min(fitnessList),step=step)
    tf.summary.scalar("Mean Scoring", np.mean(fitnessList),step=step)
    tf.summary.scalar("Variance", np.var(fitnessList),step=step)

if __name__ == "__main__":
    should_run = True
    signal.signal(signal.SIGINT, close_signal)
    init()

    for i in range(EPISODES):
        if not should_run:break #Quit condition

        print(F"Running Generation {i}")
        fitness = episode()
        log_stats(fitness,i)
        
        # Declare Elite
        if ELITE_SIZE > 0:
            elite_idx = np.argsort(-np.array(fitness))[:ELITE_SIZE]
        else:
            elite_idx = []
        
        parent_idx = choose_parents(fitness)

        new_generation = []
        # Keep elite as is
        for idx in elite_idx:
            agents[idx].id = len(new_generation)
            new_generation.append(agents[idx])
        
        for _ in range(ALIENS):
            new_generation.append(Agent(ENV,len(new_generation)+1))

        for i in range(len(parent_idx)):
            child_agent = Agent(ENV,len(new_generation)+1)
            child_agent.copy_brain(agents[parent_idx[i]])
            child_agent.mutate()
            new_generation.append(child_agent)
        
        agents = new_generation
        
        time.sleep(3) #?
