import river
import pandas as pd
import re

class HashingTrickTey:

    def __init__(self, hashRange: int = 500):
        self.hashRange = hashRange

    def createHashMatrix(self, corpus):
        documentDict = {}
        count = 0
        for sentence in corpus:
            words = self.transform_one(sentence)
            documentDict[count] = words
            count += 1
        return documentDict

    def transform_one(self, document: str):
        documentDict = {}

        for i in range(self.hashRange):
            documentDict[i] = 0

        # print(document)
        document = document.lower()
        words = re.compile(r"(?u)\b\w\w+\b").findall(document)
        # print(document)
        # words = words.lower()

        for word in words:
            documentDict[hash(word) % self.hashRange] += 1
        return documentDict

    def transform_many(self, corpus):
        data = {}
        columns = []
        matrix = self.createHashMatrix(corpus)
        for x in matrix.values():
            columns += [*x]

        columns = set(columns)

        for doc in [*matrix]:
            data[doc] = [matrix[doc][i]
                        if i in matrix[doc].keys()
                        else 0
                        for i in columns]
        return pd.DataFrame(data=data.values(),
                            columns=columns, index=matrix.keys())
    def fit(self, document):
        pass
