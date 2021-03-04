from pyopp import cMessage
import numpy as np


class dataMessage(cMessage):
    def __init__(self, name, kind = 0):
        cMessage.__init__(self,name, kind)
        self.items_embeddings = np.empty(0)
        self.user_ratings = np.empty(0)
        self.initial_weights = []


