from copy import copy


class Node(object):

    def __init__(self):
        self._left: 'Node' = None
        self._right: 'Node' = None
        self._data = None

    @property
    def left(self) -> 'Node':
        return copy(self._left)

    @property
    def right(self) -> 'Node':
        return copy(self._right)

