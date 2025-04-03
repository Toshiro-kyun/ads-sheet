"""
List of contents:

1. Tutorial Exercises
2. Lecture Code
3. Implementation of Abstract Datastructures
4. Time-complexity sheet
"""




"""
Implementation of Abstract Datastructures
"""
# Stack Implementation
class Stack:
    """
    Creates a stack that allows for last-in-first-out (LIFO) access
    """

    def __init__(self):
        self._stack = []

    def push(self, item) -> None:
        """
        Adds an item to the top of the stack
        :param item: item to add
        """
        self._stack.append(item)

    def pop(self):
        """
        Removes and returns the item on top of the stack
        :return: item on top of the stack
        """
        return self._stack.pop()

    def size(self) -> int:
        """
        Returns the number of items on the stack
        :return: number of items on the stack
        """
        return len(self._stack)

# Queue implementation, keeping track of front and back
class Queue:
    """
    Creates a first-in-first-out (FIFO) queue
    """

    def __init__(self):
        self._q = [0, 0]
        self._front = 0
        self._back = 0

    def size(self) -> int:
        """
        Returns the number of items in the queue
        :return: number of items in the queue
        """
        size = self._back - self._front
        if self._front > self._back:
            size += len(self._q)
        return size

    def dequeue(self):
        """
        Removed and returns the oldest item in the queue
        :return: the item that has been in the queue the longest
        """
        if self.size() == 0:
            print("Queue empty!")
            return None
        item = self._q[self._front]
        self._front = (self._front + 1) % len(self._q)
        return item

    def enqueue(self, item) -> None:
        """
        Adds an item to the end of the queue
        :param item: item to add to the queue
        """
        self._q[self._back] = item
        self._back = (self._back + 1) % len(self._q)
        if self._back == self._front:
            self._back += len(self._q)
            self._q.extend(self._q)


# Queue as two stacks
class Queue:
    """
    Creates a first-in-first-out (FIFO) queue
    """

    def __init__(self):
        self._stack0 = Stack()
        self._stack1 = Stack()

    def size(self) -> int:
        """
        Returns the number of items in the queue
        :return: number of items in the queue
        """
        return self._stack0.size() + self._stack1.size()

    def dequeue(self):
        """
        Removed and returns the oldest item in the queue
        :return: the item that has been in the queue the longest
        """
        if self._stack1.size() == 0:
            while self._stack0.size() > 0:
                self._stack1.push(self._stack0.pop())
        return self._stack1.pop()

    def enqueue(self, item) -> None:
        """
        Adds an item to the end of the queue
        :param item: item to add to the queue
        """
        self._stack0.push(item)

# Queue as simple list implementation
class Queue:
    """
    Creates a first-in-first-out (FIFO) queue
    """

    def __init__(self):
        self._q = []

    def size(self) -> int:
        """
        Returns the number of items in the queue
        :return: number of items in the queue
        """
        return len(self._q)

    def dequeue(self):
        """
        Removed and returns the oldest item in the queue
        :return: the item that has been in the queue the longest
        """
        return_value = self._q[0]
        del self._q[0]
        return return_value

    def enqueue(self, item) -> None:
        """
        Adds an item to the end of the queue
        :param item: item to add to the queue
        """
        self._q.append(item)
