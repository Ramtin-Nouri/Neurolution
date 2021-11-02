from Agent import Agent
import time

nAgents = 100
EPISODES = 100
ENV = "PongNoFrameskip-v4"
minimalRequiredFitness = -4

def createAgents():
    global agents
    agents = []
    for i in range(nAgents):
        agents.append(Agent(ENV,i))

def episode():
    global agents
    for agent in agents:
        agent.run()

    for i in range(len(agents)):
        agents[i].thread.join()
        print(agents[i].fitness)
        if agents[i].fitness < minimalRequiredFitness:
            agents[i] = Agent(ENV,i)
        else:
            weights = [layer.get_weights() for layer in agents[i].brain.model.layers]
            agents[i].dna.setGens(weights)
    

def mutate():
    global agents
    for agent in agents:
        agent.dna.mutate()

print("Creating Agents")
createAgents()

for i in range(EPISODES):
    print(F"Running Generation {i}")
    episode()
    #cross()
    mutate()
    time.sleep(3) #?