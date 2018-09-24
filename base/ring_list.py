class RingList:
    def __init__(self):
        self._max_size = 0
        self._ring_list = []
        self._index_to_replace = -1

    def init(self, max_size):
        self._max_size = max_size
        self.clear()

    def add(self, x):  # TODO optimize
        if len(self._ring_list) < self._max_size:  # usual adding
            self._ring_list.append(x)
        else:  # len(self._ring_list) == self._max_size  -->  ring-style adding
            self._index_to_replace += 1
            if self._index_to_replace >= self._max_size:
                self._index_to_replace = 0
            self._ring_list[self._index_to_replace] = x

    def clear(self):
        self._ring_list.clear()
        self._index_to_replace = -1

    def get(self):
        return self._ring_list

    def get_actual_size(self):
        return len(self._ring_list)

    def get_index_to_replace(self):
        return self._index_to_replace
