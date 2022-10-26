import pandas as pd
import numpy as np
import re
from space_saving import SpaceSaving
from space_saving_heap import SpaceSavingAlgorithm


class IncrementalWordContext:
    def __init__(self, num_words, num_contexts, window_size, fixed_contexts=False):
        self.__num_words = num_words
        self.__num_contexts = num_contexts
        self.__df = pd.DataFrame(np.zeros((num_words, num_contexts)))
        self.__ppmi = pd.DataFrame(np.zeros((num_words, num_contexts)))
        self.__word2idx = {}
        self.__idx2word = {}
        self.__window_size = window_size
        self.__D = 0  # Number of tokens, According to Incremental Time Evolving article.
        self.__index_word = 0
        self.__index_context = 0
        self.__word = 0
        self.__counts = {}
        self.__space_saving_words = SpaceSavingAlgorithm(num_words)
        self.__space_saving_contexts = SpaceSavingAlgorithm(num_contexts)
        self.__fixed_contexts = fixed_contexts

    def fit(self, X):
        """
        X = X.lower()
        X = X.replace("~", " ")
        X = re.sub("\s\s+", " ", X)
        X = X.strip()
        X = "".join([character for character in X if character.isalnum() or character == " "])
        X = re.sub("\d+", "0", X)
        """
        sentence = X.split(' ')
        # self.__D += len(sentence)
        self.__update_counts(sentence)
        for idx, token in enumerate(sentence):
            self.__D += 1
            window = self.__get_window(sentence, idx)

            target = sentence[idx]  # idx: target; the others: context
            for w in window:
                if w not in self.__df.columns:
                    self.__update_matrix_columns_and_rows(w)

                self.__update_cooccurrence_counts_row(target, w)
                self.__update_ppmi_row(target, w)

            # if self.__index < self.__num_words or sum([1 for i in window if i in self.__df.columns]) == len(window):
            # self.__update_cooccurrence_counts(sentence, idx, window)
            # self.__update_ppmi(sentence, idx, window)

    def __get_window(self, sentence, target_position):
        target = sentence[target_position]
        sentence_size = len(sentence)
        return sentence[max(0, target_position - self.__window_size): min(target_position + self.__window_size + 1,
                                                                          sentence_size)]

    def __update_cooccurrence_counts_row(self, target, w):
        if target in self.__df.index and w in self.__df.columns:
            self.__df.at[target, w] += 1

    def __update_cooccurrence_counts(self, sentence, target_position, window):
        target = sentence[target_position]
        sentence_size = len(sentence)
        for w in window:
            if target == w:
                continue

            # self.__df.at[w, target] += 1
            if target in self.__df.index and w in self.__df.columns:
                self.__df.at[target, w] += 1

    def __update_counts(self, sentence):
        for w in sentence:
            self.__counts[w] = self.__counts[w] + 1 if w in self.__counts else 1

    def __update_ppmi_row(self, target, w):
        if target in self.__df.index and w in self.__df.columns:
            dividend = self.__df.at[target, w] * self.__D

            divisor = self.__counts[w] * self.__counts[target]
            if divisor != 0:
                ppmi_calculated = max(0, np.log2(dividend / divisor))
            else:
                ppmi_calculated = 0

            self.__ppmi.at[target, w] = max(0, ppmi_calculated)

    def __update_ppmi(self, sentence, target_position, window):
        target = sentence[target_position]
        sentence_size = len(sentence)
        for w in window:
            if target == w:
                continue

            if target in self.__df.index and w in self.__df.columns:
                dividend = self.__df.at[target, w] * self.__D

                divisor = self.__counts[w] * self.__counts[target]
                if divisor != 0:
                    ppmi_calculated = max(0, np.log2(dividend / divisor))
                else:
                    ppmi_calculated = 0

                self.__ppmi.at[target, w] = max(0, ppmi_calculated)

    def __update_matrix_columns_and_rows(self, word):
        dumped_word = self.__space_saving_words.insert(word)

        dumped_context = None
        if not self.__fixed_contexts or not self.__space_saving_contexts.is_full():
            dumped_context = self.__space_saving_contexts.insert(word)

        if dumped_word is not None:
            self.__df.rename({dumped_word['element']: word}, axis=0, inplace=True)
            self.__ppmi.rename({dumped_word['element']: word}, axis=0, inplace=True)
            self.__df.loc[word] = np.zeros(self.__num_contexts)
            self.__ppmi.loc[word] = np.zeros(self.__num_contexts)
            # self.__word2idx[word] = self.__index
            # self.__idx2word[self.__index] = word
            # del self.__idx2word[self.__word2idx[dumped_word['element']]]
            # del self.__word2idx[dumped_word['element']]
        elif word not in self.__df.index:
            self.__df.rename({self.__index_word: word}, axis=0, inplace=True)
            self.__ppmi.rename({self.__index_word: word}, axis=0, inplace=True)
            # self.__word2idx[word] = self.__index
            # self.__idx2word[self.__index] = word
            self.__index_word += 1
            # self.__word += 1

        if dumped_context is not None and not self.__fixed_contexts:
            self.__df.rename({dumped_context['element']: word}, axis=1, inplace=True)
            self.__ppmi.rename({dumped_context['element']: word}, axis=1, inplace=True)
            self.__df[word] = 0
            self.__ppmi[word] = 0
        elif word not in self.__df.columns:
            if not self.__space_saving_contexts.is_full():
                self.__df.rename({self.__index_context: word}, axis=1, inplace=True)
                self.__ppmi.rename({self.__index_context: word}, axis=1, inplace=True)
                # self.__word2idx[word] = self.__index
                # self.__idx2word[self.__index] = word
                self.__index_context += 1
                # self.__word += 1

    def transform_one(self, sentence, for_river=True):
        ppmis = np.array([])
        for i in sentence.split(' '):
            if i in self.__ppmi.index:
                ppmis = np.vstack([ppmis, self.__ppmi.loc[i]]) if len(ppmis) != 0 else np.array([self.__ppmi.loc[i]])

        if len(ppmis) == 0:
            ppmis = np.array(np.zeros((self.__num_words, self.__num_contexts)))

        if not for_river:
            return np.mean(ppmis, axis=0)
        return {i: v for i, v in enumerate(np.mean(ppmis, axis=0))}

    @property
    def ppmi_matrix(self):
        return self.__ppmi

    @property
    def word_context_matrix(self):
        return self.__df

    @property
    def word_counts(self):
        return self.__counts