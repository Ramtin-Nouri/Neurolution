from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Softmax, Dense, Flatten
import numpy as np
from pygad import kerasga
from Config import MODELTYPE

class Brain():
    def __init__(self, inputShape, nActions):
        self.model = Sequential()
        if MODELTYPE == "CNN":
            self.createModel2D(inputShape, nActions)
        else:
            self.createModel(inputShape, nActions)

    def createModel2D(self, inputShape, nActions):
        self.model.add(Conv2D(32, (3, 3), activation='relu',padding='same',input_shape=inputShape))
        self.model.add((MaxPooling2D(2,2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu',padding='same'))
        self.model.add((MaxPooling2D(2,2)))
        self.model.add(Flatten())
        self.model.add(Dense(32,activation="sigmoid"))
        self.model.add(Dense(nActions,activation="sigmoid"))
        self.model.add(Softmax())

    def createModel(self, inputShape, nActions):
        self.model.add(Dense(128,activation="sigmoid",input_shape=inputShape))
        self.model.add(Dense(nActions,activation="sigmoid"))
        self.model.add(Softmax())


    def decide(self,state):
        return np.argmax(self.model.predict(np.expand_dims(state,axis=0)))

    def create_brain_from_dna(self, dna):
        kerasga.model_weights_as_matrix(self.model, dna)