from sentence_transformers import SentenceTransformer
import pandas as pd

class BertEy:
    def __init__(self):
        self = self
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

    def transform_many(self, sentences):
        data = {}
        indexes = []
        columns = [x for x in range(384)]
        for index, sentence in enumerate(sentences):
            data[index] = self.transform_one(sentence)
            indexes.append(index)

        return pd.DataFrame(data=data.values(),
                            columns=columns, index=indexes)

    def transform_one(self, document):
        sentence_embedding = self.model.encode(document)
        teste = {}
        for count, value in enumerate([embed for embed in sentence_embedding]):
            teste[count] = value
        return teste

    def fit(self, document):
        pass
        # splited = [document.split(" ") for i in range(2)]
        # self.model.train(splited, total_examples=1, epochs=1)