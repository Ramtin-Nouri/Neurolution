import numpy as np
from tqdm import tqdm
from threading import Thread
import gym
from DNA import DNA
from Model import Brain
from Config import MAX_SIM_STEPS


# TODO: set seeds?
class Agent(Thread):
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
        self.env = gym.make(envName)
        shapeIn = list(self.env.observation_space.shape)
        shapeOut = self.env.action_space.n
        self.brain = Brain(shapeIn, shapeOut)
        self.dna = DNA(self.brain)
        self.fitness = 0
        self.thread = None
        self.id = id

    def run(self):
        self.thread = Thread(target=self.simulate)
        self.thread.start()

    def simulate(self):
        done = False
        self.fitness = 0
        obs = self.env.reset()
        nSteps = 0
        if self.id==0:
            for _ in tqdm(range(MAX_SIM_STEPS)):
                if done:break
                obs, reward, done, _ =  self.env.step(self.get_action(obs))
                self.fitness += reward
                nSteps += 1
        else:
            while not done and nSteps < MAX_SIM_STEPS:
                obs, reward, done, _ =  self.env.step(self.get_action(obs))
                self.fitness += reward
                nSteps += 1
                #if self.id==0:self.env.render()

    def get_action(self,state):
        return self.brain.decide(state)
    
    def mutate(self):
        self.dna.get_dna()
        self.dna.mutate()
        self.brain.create_brain_from_dna(self.dna.vector)

    def copy_brain(self, otherAgent):
        otherAgent.dna.get_dna()
        self.dna.vector = otherAgent.dna.vector.copy()
        self.dna.set_dna()
