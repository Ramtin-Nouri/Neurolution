from tqdm import tqdm
from threading import Thread
import gym
from DNA import DNA
from Model import Brain
from Config import MAX_SIM_STEPS, WRAP_ENV,ENV
import Environment

# TODO: set seeds?
class Agent():
    """
    Agent

    Attributes:
    env: OpenAiGym Environment 
        Every Agent has his own environment for simulation
        S.t. they can all simulate in parallel
    dna: DNA (See DNA.py)
    brain: Brain (See Brain.py)
    fitness: float
        current fitness score
    thread: Thread
        thread to run simulation
    id: int
    """
    def __init__(self, envName,id):
        if WRAP_ENV:
            self.env = Environment.EnvWrapper(ENV,True)
        else:
            self.env = gym.make(envName)
        shapeIn = list(self.env.observation_space.shape)
        shapeOut = self.env.action_space.n
        self.brain = Brain(shapeIn, shapeOut)
        self.dna = DNA(self.brain)
        self.fitness = 0
        self.thread = None
        self.id = id
        self.should_run = True

    def run(self):
        """Start thread"""
        self.thread = Thread(target=self.simulate)
        self.thread.start()

    def simulate(self):
        """Thread loop"""
        done = False
        self.fitness = 0
        obs = self.env.reset()
        
        if self.id==0: # if id==0 print progressbar
            loopRange = tqdm(range(MAX_SIM_STEPS))
        else:
            loopRange = range(MAX_SIM_STEPS)
            
        for _ in loopRange:
            if done or not self.should_run :break
            obs, reward, done, _ =  self.env.step(self.get_action(obs))
            self.fitness += reward
        # simulation finished. Print results
        print(F"Fitness of agent {self.id}: {self.fitness}")

    def get_action(self,state):
        """Return (neural network's choice of) best action given game state"""
        return self.brain.decide(state)
    
    def mutate(self):
        """Mutate DNA"""
        self.dna.get_dna()
        self.dna.mutate()
        self.brain.create_brain_from_dna(self.dna.vector)

    def copy_brain(self, otherAgent):
        """Copy DNA of given Agent to this agent's DNA"""
        otherAgent.dna.get_dna()
        self.dna.vector = otherAgent.dna.vector.copy()
        self.dna.set_dna()
