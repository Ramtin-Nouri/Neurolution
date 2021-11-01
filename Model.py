from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Softmax, Dense, Flatten
import numpy as np
from DNA import DNA

class Brain():
    def __init__(self, dna, inputShape, nActions):
        self.model = Sequential()
        self.createModel(inputShape, nActions)

    def createModel(self, inputShape, nActions):
        self.model.add(Conv2D(8, (3, 3), activation='relu',padding='same',input_shape=inputShape))
        self.model.add((MaxPooling2D(2,2)))
        self.model.add(Conv2D(16, (3, 3), activation='relu',padding='same'))
        self.model.add((MaxPooling2D(2,2)))
        self.model.add(Conv2D(32, (3, 3), activation='relu',padding='same'))
        self.model.add((MaxPooling2D(2,2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
        self.model.add((MaxPooling2D(2,2)))
        self.model.add(Flatten())
        self.model.add(Dense(128,activation="sigmoid"))
        self.model.add(Dense(nActions,activation="sigmoid"))
        self.model.add(Softmax())

    def decide(self,state):
        return np.argmax(self.model.predict(np.expand_dims(state,axis=0)))