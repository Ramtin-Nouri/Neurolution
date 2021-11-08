import numpy as np
from threading import Thread
import gym
from DNA import DNA
from Model import Brain

MAX_STEPS = 1000

# TODO: set seeds?
class Agent(Thread):
    """
    Agent

    Attributes:
    env: OpenAiGym Environment 
        Every Agent has his own environment for simulation
        S.t. they can all simulate in parallel
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
        while not done and nSteps < MAX_STEPS:
            obs, reward, done, _ =  self.env.step(self.getAction(obs))
            self.fitness += reward
            nSteps += 1
            #if self.id==0:self.env.render()

    def getAction(self,state):
        return self.brain.decide(state)
    
    def mutate(self):
        self.dna.mutate()
        self.brain.create_brain_from_dna(self.dna.vector)
