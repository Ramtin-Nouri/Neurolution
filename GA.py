from Agent import Agent
import time
# No idead why but I need this to not get an error:
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)



nAgents = 10
EPISODES = 1
ENV = "PongNoFrameskip-v4"

def create_agents():
    global agents
    agents = []
    for i in range(nAgents):
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
    

def mutate():
    global agents
    for agent in agents:
        agent.dna.mutate()

def choose_parents(fitness):
    pass

print("Creating Agents")
create_agents()

for i in range(EPISODES):
    print(F"Running Generation {i}")
    fitness = episode()
    choose_parents(fitness)
    #cross()
    mutate()
    time.sleep(3) #?
