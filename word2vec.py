from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec, KeyedVectors
import gensim.downloader
import pandas as pd
import numpy as np

class Word2VecTey:
    size: int
    model: Word2Vec

    def __init__(self, size=100):
        self.size = size
        #wiki-english-20171001
        self.model = Word2Vec(sentences=common_texts, size=100, window=5, min_count=1, workers=4, \
        batch_words=1000)

    def transform_many(self, phrases):
        dictTey = {}
        indexes = []
        columns = [x for x in range(self.size)]
        for index, phrase in enumerate(phrases):
            dictTey[index] = self.transform_one(phrase, self.size)
            indexes.append(index)

        return pd.DataFrame(data=dictTey.values(),
                            columns=columns, index=indexes)

    def transform_one(self, document):

        dictToBeReturned = {}

        # self.fit(document)
        represent = None

        for p in document.split(" "):
            v = np.array(self.model.wv[p])
            if represent is None:
                represent = v
            else:
                represent = np.add(v, represent)

        represent /= len(document.split(" "))

        for count, value in enumerate(represent):
            dictToBeReturned[count] = value
        return dictToBeReturned

    def fit(self, document):
        splited_document = document.split(" ")

        # print("Mestora document:", splited_document,
        #       "of size:", len(splited_document),
        #       "of type:", type(splited_document))

        # Based on below reference, train doesn't actually updates model's vocabullary
        # https://stackoverflow.com/questions/55774197/gensims-word2vec-not-training-provided-documents

        # Solved based on
        # https://stackoverflow.com/questions/42357678/gensim-word2vec-array-dimensions-in-updating-with-online-word-embedding
        self.model.build_vocab([splited_document], update=True)
        self.model.train([splited_document], total_examples=1, epochs=1)