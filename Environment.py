# from https://github.com/Ramtin-Nouri/TF2_Keras_Reinforce/blob/main/dataManager.py
import os,random,cv2,numpy as np
import gym

# preprocessing used by Karpathy (cf. https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5)
def preprocess_frame_karpathy(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).reshape(80*80) #TODO: make dependent of stack


class EnvWrapper():
    """
        Wrapper around actual OpenAi Gym environent with some processing of the observations
    """
    def __init__(self,env_id,use_preprocessing,use_diff=True,stack=False):
        """
        Arguments
        ---------
        env_id: str
            Name of the OpenAi Gym Environment
        use_preprocessing: bool
            Whether observations should be simplified with preprocess_frame_karpathy or not
        use_diff:
            Whether the last observation should be subtracted from the current and this results in the outputted observation
        stack:
            Whether the last and current observation should be stacked as the outputted observation.
        """
        self.env = gym.make(env_id).env
        self.use_preprocessing = use_preprocessing
        self.use_diff = use_diff
        self.lastObservation = None
        self.stack = stack

        
        class ActionSpace():
            def __init__(self,n):
                self.n = n

        self.action_space = ActionSpace(self.env.action_space.n)

        class ObservationSpace():
            def __init__(self,s):
                self.shape = s

        if stack:
            if use_preprocessing:
                self.observation_space = ObservationSpace((2,80*80))
            else:
                self.observation_space = ObservationSpace((2,210,160,3))
        else:
            self.observation_space = ObservationSpace([80*80])
    
    def reset(self):
        """
            If use_preprocess is True the outputted observation will run through preprocess_frame_karpathy
            else it will only be normalized 
            and then returned
            
            Returns
            -------
            observation: np.array()
                (preprocessed) initial state observation 
        """
        observation = self.env.reset()
        if self.use_preprocessing:
            observation = preprocess_frame_karpathy(observation)
        else:
            observation = np.array(observation)/255
        if self.use_diff or self.stack:
            self.lastObservation = observation
        return observation
    
    def step(self,action):
        """
            The returned observation of env.step() will be handled the same as reset()
            At last it will also be subtracted by the last observation before being returned (if use_diff is set to True).
            The other returned values from env.step remain unchanged
            
            Arguments
            --------
            action: int
                See OpenAi Gym Documentation
            
            Returns
            -------
            diff: np.array()
                difference between current (processed) observation and last observation
            For description of returns See OpenAi Gym Documentation
            
            
        """
        observation, reward, done, info = self.env.step(action)
        if self.use_preprocessing:
            observation = preprocess_frame_karpathy(observation)
        else:
            observation = np.array(observation)/255
        if not self.use_diff:
            return observation,reward,done,info
        if self.stack:
            stacked = np.stack((observation,self.lastObservation))
            self.lastObservation = observation
            return stacked ,reward, done, info 
        diff = observation-self.lastObservation
        self.lastObservation = observation
        return diff, reward, done, info