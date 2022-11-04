import river
import pandas as pd
import numpy as np
import re
import os

class Glove:

    def __init__(self, dimensions=300):
        self.embeddings_index = {}
        self.dimensions = dimensions
        
        filename = f"pre-training/glove.6B.{dimensions}d.txt"

        with open(filename) as f:
            for line in f:
                values = line.split(' ')
                word = values[0] 
                coefs = np.asarray(values[1:], dtype='float32') 
                self.embeddings_index[word] = coefs

    def transform_one(self, document: str):
        dictToBeReturned = {}
        represent = None

        for p in document.split(" "):
            if p in self.embeddings_index:
                v = np.array(self.embeddings_index[p])
                if represent is None:
                    represent = v
                else:
                    represent = np.add(v, represent)

        if represent is not None:
            represent /= len(document.split(" "))
        else:
            represent = np.zeros(self.dimensions)

        for count, value in enumerate(represent):
            dictToBeReturned[count] = value
        return dictToBeReturned

    def transform_many(self, corpus):
        pass

    def fit(self, document):
        pass