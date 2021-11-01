from Agent import Agent

nAgents = 5
EPISODES = 2
ENV = "PongNoFrameskip-v4"

def createAgents():
    global agents
    agents = []
    for i in range(nAgents):
        agents.append(Agent(ENV,i))

def episode():
    global agents
    for agent in agents:
        agent.run()

    for agent in agents:
        agent.thread.join()
        print(agent.fitness)
       
    
print("Creating Agents")
createAgents()

for i in range(EPISODES):
    print(F"Running Episode {i}")
    episode()