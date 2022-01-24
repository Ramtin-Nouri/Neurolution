# Neurolution
A Neuro-Evolution implementation. Developed for the Uni-Hamburg *bio-inspired artificial intelligence* seminar.

# About 
Evolves a neural network (implemented in tensprflow.keras) to learn to play Pong (using the OpenAI Gym environment). I implement 4 different mutation types and crossover.  
The parents of the next generation are chosen using the roulette wheel selection. Also the $n$ best scoring individuals are always kept unchanged for the next generation. $n$ can be changed in the config.py under ELITE_SIZE. ALIENS defines the number of totally random individuals that are additionally also added to each generation.

Two models are defined. A dense network and a convolutional one.

# Dependencies
- numpy 
- tensorflow2
- pygad
- Openai Gym[atari]

# Execute
- update the config.py with your desired values
- run ```python3 GA.py```