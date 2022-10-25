import heapq
from typing import Counter


class SpaceSavingAlgorithm:
    """
    Efficient `Counter`-like structure for approximating the top `m` elements of a stream, in O(m)
    space (https://www.cse.ust.hk/~raywong/comp5331/References/EfficientComputationOfFrequentAndTop-kElementsInDataStreams.pdf).

    Specifically, the resulting counter will contain the correct counts for the top k elements with
    k ≈ m.  The interface is the same as `collections.Counter`.
    """

    def __init__(self, m):
        self._m = m
        self._elements_seen = 0
        self._counts = Counter()  # contains the counts for all elements
        self._queue = []  # contains the estimated hits for the counted elements

    def insert(self, x):
        self._elements_seen += 1

        if x in self._counts:
            self._counts[x] += 1
            return None
        elif len(self._counts) < self._m:
            self._counts[x] = 1
            self._heappush(1, self._elements_seen, x)
            return None
        else:
            return {'element': self._replace_least_element(x)}

    def _replace_least_element(self, e):
        while True:
            count, tstamp, key = self._heappop()
            assert self._counts[key] >= count

            if self._counts[key] == count:
                del self._counts[key]
                count = 0
                break
            else:
                self._heappush(self._counts[key], tstamp, key)

        self._counts[e] = count + 1
        self._heappush(count, self._elements_seen, e)
        return key

    def _heappush(self, count, tstamp, key):
        heapq.heappush(self._queue, (count, tstamp, key))

    def _heappop(self):
        return heapq.heappop(self._queue)

    def most_common(self, n=None):
        return self._counts.most_common(n)

    def elements(self):
        return self._counts.elements()

    def is_full(self):
        return len(self._counts) == self._m

    def __len__(self):
        return len(self._counts)

    def __getitem__(self, key):
        return self._counts[key]

    def __iter__(self):
        return iter(self._counts)

    def __contains__(self, item):
        return item in self._counts

    def __reversed__(self):
        return reversed(self._counts)

    def items(self):
        return self._counts.items()

    def keys(self):
        return self._counts.keys()

    def values(self):
        return self._counts.values()

    def update(self, iter):
        for e in iter:
            self.insert(e)