import copy
import time


class SpaceSaving:
    def __init__(self, size):
        self.size = size
        self.structure = []
        self.elements = {}

    def insert(self, element):
        dumped = None
        if not self.is_full() and element not in self.elements:
            self.structure.append({'element': element, 'count': 1, 'age': -1})
            self.elements[element] = len(self.structure) - 1
        elif self.is_full() and element not in self.elements:
            del self.elements[self.structure[-1]['element']]
            self.elements[element] = len(self.structure) - 1
            dumped = copy.deepcopy(self.structure[-1])
            self.structure[-1] = {'element': element, 'count': self.structure[-1]['count'] + 1, 'age': -1}
        elif element in self.elements:
            self.structure[self.elements[element]]['count'] += 1

        for element in self.elements.items():
            self.structure[element[1]]['age'] += 1

        self.structure = sorted(self.structure, key=lambda x: (x['count'], - x['age']), reverse=True)
        self.elements = {j['element']: i for i, j in enumerate(self.structure)}

        return dumped

    def is_full(self):
        return len(self.structure) == self.size
