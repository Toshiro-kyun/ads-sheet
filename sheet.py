"""
List of contents:

1. Tutorial Exercises
2. Implementation of Abstract Datastructures
3. Time-complexity sheet
4. Some commen problems with their solutions (implemented by Mohamad)


During the exam remember that you also have access to the reader.
"""

'''
TO DO:
Matrix representation of graphs
Pseudocode for everything
Lecture code
Time complexity sheet
Compressed, compact and suffix tries
Grammar
A* algorithm
'''

"""
Implementation of Abstract Datastructures
"""

'''
What is a stack?
A stack is a data structure that allows for two operations: we can add items onto the stack
using the push method and remove the most recently added item using the pop method.
'''
# Stack Implementation
class Stack:
    """
    Creates a stack that allows for last-in-first-out (LIFO) access
    """

    def __init__(self):
        self._stack = []

    def push(self, item) -> None: #(o(1))
        """
        Adds an item to the top of the stack
        :param item: item to add
        """
        self._stack.append(item)

    def pop(self): #(o(1))
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
        
    def peek(self):
        if not self.stack:
            print("Stack is empty")
            return None
        return self.stack[-1]

    def search(self, value):
        for i in range(len(self.stack)-1, -1, -1):  #Search from top to bottom
            if self.stack[i] == value:
                return len(self.stack) - 1 - i  #Distance from top
        print("Value not found")
        return -1

    def remove_max_from_stack(stack):
    if not stack:
        print("Stack is empty")
        return

    temp = []
    max_val = max(stack)

    #Move to temp and skip first max
    removed = False
    while stack:
        val = stack.pop()
        if val == max_val and not removed:
            removed = True
            continue
        temp.append(val)

    #Restore stack
    while temp:
        stack.append(temp.pop())

    def remove_min_from_stack(stack):
        if not stack:
            print("Stack is empty")
            return
    
        temp = []
        min_val = min(stack)
    
        removed = False
        while stack:
            val = stack.pop()
            if val == min_val and not removed:
                removed = True
                continue
            temp.append(val)
    
        while temp:
            stack.append(temp.pop())


# Queue implementation, keeping track of front and back
'''
A queue is a linear data structure that follows the First-In-First-Out (FIFO) principle, 
where the first element added to the queue is the first one to be removed.
'''
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

    def dequeue(self): #(o(1))
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

    def enqueue(self, item) -> None: #(o(1))
        """
        Adds an item to the end of the queue
        :param item: item to add to the queue
        """
        self._q[self._back] = item
        self._back = (self._back + 1) % len(self._q)
        if self._back == self._front:
            self._back += len(self._q)
            self._q.extend(self._q)

    def peek(self):
        if not self.queue:
            print("Queue is empty")
            return None
        return self.queue[0]

    def search(self, value):
        for i in range(len(self.queue)):
            if self.queue[i] == value:
                return i  #Position from front
        print("Value not found")
        return -1

    def remove_max_from_queue(queue):
    if not queue:
        print("Queue is empty")
        return

    max_val = max(queue)
    new_queue = []
    removed = False
    for val in queue:
        if val == max_val and not removed:
            removed = True
            continue
        new_queue.append(val)

    queue[:] = new_queue 

    def remove_min_from_queue(queue):
        if not queue:
            print("Queue is empty")
            return
    
        min_val = min(queue)
        new_queue = []
        removed = False
        for val in queue:
            if val == min_val and not removed:
                removed = True
                continue
            new_queue.append(val)
    
        queue[:] = new_queue


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

#Linked Lists (done by chirag) 

class LinkedList:
    """
    Creating a linked list data structure
    """
    def __init__(self, item = None, next = None):
        self.item = item
        self.next = next

    def list_empty_error(self) -> None:
        print("List empty")

    def get_first_item(self): #(o(1))
        """
        Returns the first value stored in the linked list
        :return: the first value stored in the linked list
        """
        if self.next is None:
            self.list_empty_error()
            return None
        return self.next.item

    def remove_first_item(self): 
        """
        Returns and removes the first value stored in the linked list
        :return: the first value stored in the linked list
        """
        if self.next is None:
            self.list_empty_error()
            return None
        item = self.next.item
        self.next = self.next.next
        return item

    def size(self) -> int: #(o(n))
        """
        Returns the number of items stored in the linked list
        :return: the number of items stored in the linked list
        """
        size = 0
        node = self.next
        while node is not None:
            size += 1
            node = node.next
        return size

    def visit(self, action) -> None:
        """
        Performs an action for every item stored on the list
        :param action: function that takes the value of a node as input
        """
        node = self.next
        while node is not None:
            action(node.item)
            node = node.next

    def invalid_position(self) -> None:
        print("Invalid position")

    def list_too_short_error(self) -> None:
        print("Linked list is too short")

    def get_item(self, curr_pos: int):
        """
        Returns the item at the given position in the linked list
        :param curr_pos: position of the item to return
        :return: the item at the given position in the linked list
        """
        if curr_pos < 0:
            self.invalid_position()
            return None
        node = self.next
        for _ in range(curr_pos):
            if node is None:
                self.list_too_short_error()
                return None
            node = node.next
        return node.item if node else None

    def add(self, item, curr_pos: int = 0) -> None:
        """
        Adds an item to the linked list
        :param item: item to add
        :param curr_pos: position to add the item (by default at the start of the linked list)
        """
        if curr_pos < 0:
            self.invalid_position()
        elif curr_pos == 0:
            self.next = LinkedList(item, self.next)
        elif self.next is not None:
            self.next.add(item, curr_pos - 1)
        else:
            self.list_too_short_error()

    def value_not_found_error(self):
        print("Value not found")

    def remove(self, value) -> None:
        """
        Removes the first occurrence of a given value
        :param value: value to remove
        """
        if self.next is not None:
            if self.next.item == value:
                self.remove_first_item()
            else:
                self.next.remove(value)
        else:
            self.value_not_found_error()

    def remove_duplicates(self):
        current_node = self

        while current_node.next is not None:
            if current_node.next.item == current_node.item:
                current_node.next = current_node.next.next
            else:
                current_node = current_node.next
                
    def remove_min(self): #(o(n))
    if self.next is None:
        self.list_empty_error()
        return

    #Find min value
    min_val = self.next.item
    node = self.next
    while node is not None:
        if node.item < min_val:
            min_val = node.item
        node = node.next

    #Remove first occurrence of min value
    self.remove(min_val)

    def remove_max(self): #(o(n))
    if self.next is None:
        self.list_empty_error()
        return

    #Find max value
    max_val = self.next.item
    node = self.next
    while node is not None:
        if node.item > max_val:
            max_val = node.item
        node = node.next

    #Remove first occurrence of max value
    self.remove(max_val)


#Trees:
'''
What is a tree?
A tree is a data strcuture that is higherarchical. It has root nodes and children that come from those nodes.

the number of edges equals the number of nodes − 1. (you have to make sure that there are no cycling in the tree) 

What types of trees are there?

- Binary tree: Every root node has at most 2 children node. One to the right and one to the left
    - Expression trees: A tree whose leaf nodes are symbols or numbers, and the non-leaf nodes are operators. Used to calculate stuff when an traversal is applied. They remove ambuiguity from calculations.
    - Search trees:A search tree is a tree that minimises the search time for a node. They have the following charachteristics:
        -A search tree is always in order. This means that the left has to be smaller than the root, and the right has to be bigger
        -There is no repetition in the values that the tree holds.
- Heaps: Like a search tree, a heap is a binary tree where every node contains a value from a linearly 
         ordered set of values. Moreover, a heap is always a complete binary tree (meaning that its balanced) and satisfies the heap property.
         -Heap property: Every child of a node has to be equal or less than it.
- Tries: A trie is a type of tree that isnt necessarialy binary. Each node represents a letter, words are created by traveling down the tree. Prefixes are shared therefore lookup is fast
        - Standard tries have no level of compresion. If a word is 100 a in a row, it will have 100 nodes that contain a in a row.
        - Compressed tries have some compression. If all of the nodes under another node have one or zero children, their value is added to the one ontop until a situation where it cant be compressed further.
            -So if you had 100 a the node in that tree would have value aaaa...
        - The compact trie is obtained from the compressed trie by replacing the strings in the nodes by their coordinates. Basically, my list [a,b,c,d] would become [0,1,2,3] and if a word was abc, it would store 0-2.
            - every node except the root contains two numbers referring to a string;
            - the children of a node are ordered alphabetically on initial letter;
            - there are no nodes with branching degree 1;
            - the branches from the root correspond exactly with the words in W.
        - Suffix tries: Every substring of a string S is the prefix of a suffix of S. A suffix trie is a special trie that stores all the suffixes of a given string.

Trees can be:
    Balanced         # The left and right subtrees of every node differ in height by no more than 1.
    Complete         # All levels are fully filled except possibly the last, which is filled from left to right. This also means that the number of nodes is equal to the maximum index in the tree
    Full             # Every node has either 0 or 2 children — no nodes have only one child.
    Perfect          # A tree where all internal nodes have two children and all leaves are at the same level.
    Degenerate       # A tree where each parent has only one child — essentially becoming a linked list.
    Binary Search Tree (BST)   # A binary tree with the ordering property: left < root < right.
    AVL Tree         # A self-balancing BST where the difference in height between left and right subtrees is at most 1.
    Red-Black Tree   # A balanced BST with coloring rules to ensure O(log n) operations.
    B-Tree           # A self-balancing general tree used in databases and file systems — nodes can have many children.
    B+ Tree          # A variation of B-Tree where all data is stored in leaf nodes and linked for fast range queries.
    N-ary Tree       # A general tree where each node can have up to N children.
    Segment Tree     # A tree used for storing intervals or segments and answering range queries efficiently.
    Fenwick Tree (Binary Indexed Tree)   # Used for cumulative frequency/range sum queries in O(log n) time.
    Trie            # A prefix tree used for string storage and fast lookup.
    Suffix Trie     # A trie that contains all suffixes of a given string — used in string matching problems.
    Ternary Tree     # Each node has up to three children (often used in ternary search).
    Decision Tree    # A tree used in machine learning for decision making — each node is a decision point.
    Game Tree        # Represents possible moves in a game — used in AI (like chess engines).
    Parse Tree       # Represents syntactic structure of a string according to a grammar — used in compilers.
    Syntax Tree (AST)# Abstract version of a parse tree, often more compact — used in programming language processing.
    Quad Tree        # Tree where each node has 4 children — used in image processing, spatial indexing.
    Octree           # Tree with 8 children per node — used in 3D graphics and spatial partitioning.  
'''
#Basic binary tree node with no methods:
class BinaryTreeNode():
    def __init__(self, item=None, left=None, right=None):
        self._item = item
        self._left = left
        self._right = right

## Basic binary tree in node representation with functions that rebalance it and such:
# Can also be used as a search tree using "Add in search tree".
class TreeNode:
    def __init__(self, item=None, left=None, right=None):
        self._item = item
        self._height = 1
        self._left = left
        self._right = right

    def _update_height(self) -> None:
        self._height = 1
        if self._left is not None:
            self._height = max(self._height, self._left._height + 1)
        if self._right is not None:
            self._height = max(self._height, self._right._height + 1)

    def _get_height_imbalance(self) -> int:
        imbalance = 0
        if self._left is not None:
            imbalance -= self._left._height
        if self._right is not None:
            imbalance += self._right._height
        return imbalance

    def preorder(self, action) -> None:
        """
        Performs an action for each node in the tree in preorder
        :param action: function to call for every value in the tree
        """
        action(self._item)
        if self._left is not None:
            self._left.preorder(action)
        if self._right is not None:
            self._right.preorder(action)

    def postorder(self, action) -> None:
        """
        Performs an action for each node in the tree in postorder
        :param action: function to call for every value in the tree
        """
        if self._left is not None:
            self._left.postorder(action)
        if self._right is not None:
            self._right.postorder(action)
        action(self._item)

    def inorder(self, action) -> None:
        """
        Performs an action for each node in the tree in inorder
        :param action: function to call for every value in the tree
        """
        if self._left is not None:
            self._left.inorder(action)
        action(self._item)
        if self._right is not None:
            self._right.inorder(action)
    
    def postorder_node(self,action) -> None:
        """
        Performs an action for each node in the tree in postorder
        :param action: function to call for every value in the tree
        """
        if self._left is not None:
            self._left.postorder()
        if self._right is not None:
            self._right.postorder()
        action

def search_in_search_tree(tree: TreeNode, value) -> TreeNode|None:
    """
    Returns the node containing the specified value in a search tree
    :param tree: root of the search tree to search through
    :param value: value to find
    :return: node that contains the value, or None if the value is not found
    """
    if tree is None:
        return tree
    if value == tree._item:
        return tree
    if value < tree._item:
        return search_in_search_tree(tree._left, value)
    return search_in_search_tree(tree._right, value)

def add_in_search_tree(tree: TreeNode, value) -> TreeNode:
    """
    Adds a node at the correct position to preserve search tree
    :param tree: root of the search tree
    :param value: value to add
    :return: search tree with the node added
    """
    if tree is None:
        return TreeNode(value)
    if value < tree._item:
        tree._left = add_in_search_tree(tree._left, value)
        tree._update_height()
    elif value > tree._item:
        tree._right = add_in_search_tree(tree._right, value)
        tree._update_height()
    return _rebalance_tree(tree)

def _rotate_anticlockwise(tree: TreeNode) -> TreeNode:
    new_root = tree._right
    tree._right = new_root._left
    new_root._left = tree
    tree._update_height()
    new_root._update_height()
    return new_root

def _rotate_clockwise(tree: TreeNode) -> TreeNode:
    new_root = tree._left
    tree._left = new_root._right
    new_root._right = tree
    tree._update_height()
    new_root._update_height()
    return new_root

def _rebalance_tree(tree: TreeNode) -> TreeNode:
    imbalance = tree._get_height_imbalance()
    if imbalance <= -2:
        if tree._left._get_height_imbalance() <= 0:
            return _rotate_clockwise(tree)
        else:
            tree._left = _rotate_anticlockwise(tree._left)
            return _rotate_clockwise(tree)
    elif imbalance >= 2:
        if tree._right._get_height_imbalance() >= 0:
            return _rotate_anticlockwise(tree)
        else:
            tree._right = _rotate_clockwise(tree._right)
            return _rotate_anticlockwise(tree)
    return tree


#IMPORTANT, the code for the heaps was done with our best friend CHAT. So be carefull!!!!

class MinHeap:
    """
    A standard Min-Heap implementation in Python.
    
    The heap is represented as a list, where the parent node at index i has:
        - Left child at index 2*i + 1
        - Right child at index 2*i + 2

    Operations:
        - insert(value): Adds a new value to the heap while maintaining the heap property.
        - delete_min(): Removes and returns the smallest element (root) from the heap.
        - up_heap(index): Moves an element up to restore the heap property (used after insert).
        - down_heap(index): Moves an element down to restore the heap property (used after deletion).
        - peek(): Returns the smallest element without removing it.
        - size(): Returns the number of elements in the heap.
    """

    def __init__(self):
        """Initialize an empty heap."""
        self.heap = []

    def insert(self, value):
        """
        Adds a new value to the heap and restores heap order by calling up_heap.
        Time Complexity: O(log n)
        """
        self.heap.append(value)  # Add the new element to the end
        self.up_heap(len(self.heap) - 1)  # Restore heap order

    def delete_min(self):
        """
        Removes and returns the smallest element (root) from the heap.
        Replaces root with the last element and restores order using down_heap.
        Time Complexity: O(log n)
        """
        if not self.heap:
            raise IndexError("Heap is empty.")
        
        if len(self.heap) == 1:
            return self.heap.pop()

        min_value = self.heap[0]
        self.heap[0] = self.heap.pop()  # Move last element to root
        self.down_heap(0)  # Restore heap order
        return min_value

    def up_heap(self, index):
        """
        Moves an element up to maintain heap property.
        Swaps with its parent if it is smaller than the parent.
        Time Complexity: O(log n)
        """
        parent = (index - 1) // 2

        while index > 0 and self.heap[index] < self.heap[parent]:
            self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
            index = parent
            parent = (index - 1) // 2

    def down_heap(self, index):
        """
        Moves an element down to maintain heap property.
        Swaps with the smaller child if it is greater than that child.
        Time Complexity: O(log n)
        """
        size = len(self.heap)
        while True:
            left = 2 * index + 1
            right = 2 * index + 2
            smallest = index  # Assume current node is smallest

            if left < size and self.heap[left] < self.heap[smallest]:
                smallest = left

            if right < size and self.heap[right] < self.heap[smallest]:
                smallest = right

            if smallest == index:
                break  # Heap order is restored

            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            index = smallest

    def peek(self):
        """
        Returns the smallest element (root) without removing it.
        Time Complexity: O(1)
        """
        if not self.heap:
            raise IndexError("Heap is empty.")
        return self.heap[0]

    def size(self):
        """
        Returns the number of elements in the heap.
        Time Complexity: O(1)
        """
        return len(self.heap)

    def __str__(self):
        """Returns a string representation of the heap."""
        return str(self.heap)

class MinHeapNode:
    """
    A fully node-based Min-Heap.
    
    Operations:
    - insert(value): Adds a new node while maintaining heap order.
    - delete_min(): Removes the smallest value and restores order.
    - up_heap(node): Moves a node up if it violates heap property.
    - down_heap(node): Moves a node down to maintain heap property.
    - find_last_node(): Finds the last node in level order for deletions.
    """

    def __init__(self):
        self.root = None
        self.size = 0

    def insert(self, value):
        """Insert a new value into the heap."""
        new_node = MinHeapNode(value)
        if not self.root:
            self.root = new_node
            self.size = 1
            return
        
        # Find the next available parent for insertion (level order)
        parent = self.find_insertion_parent()
        if not parent.left:
            parent.left = new_node
        else:
            parent.right = new_node
        new_node.parent = parent
        self.size += 1

        self.up_heap(new_node)

    def delete_min(self):
        """Removes the smallest element (root) and restores heap order."""
        if not self.root:
            return None
        
        min_value = self.root.value
        if self.size == 1:
            self.root = None
            self.size = 0
            return min_value

        last_node = self.find_last_node()
        self.swap_values(self.root, last_node)

        # Remove the last node
        if last_node.parent:
            if last_node.parent.right == last_node:
                last_node.parent.right = None
            else:
                last_node.parent.left = None
        
        self.size -= 1
        self.down_heap(self.root)
        
        return min_value

    def up_heap(self, node):
        """Moves a node up the tree if necessary."""
        while node.parent and node.value < node.parent.value:
            self.swap_values(node, node.parent)
            node = node.parent

    def down_heap(self, node):
        """Moves a node down to maintain heap order."""
        while node.left:
            smallest = node.left
            if node.right and node.right.value < node.left.value:
                smallest = node.right
            
            if node.value <= smallest.value:
                break
            
            self.swap_values(node, smallest)
            node = smallest

    def find_insertion_parent(self):
        """Finds the next available parent node for insertion using BFS."""
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            if not node.left or not node.right:
                return node
            queue.append(node.left)
            queue.append(node.right)

    def find_last_node(self):
        """Finds the last inserted node using BFS."""
        queue = [self.root]
        last = None
        while queue:
            last = queue.pop(0)
            if last.left:
                queue.append(last.left)
            if last.right:
                queue.append(last.right)
        return last

    def swap_values(self, node1, node2):
        """Swaps values of two nodes."""
        node1.value, node2.value = node2.value, node1.value

    def peek(self):
        """Returns the smallest value without removing it."""
        return self.root.value if self.root else None

    def __str__(self):
        """Returns a level-order string representation of the heap."""
        result = []
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            result.append(node.value)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return str(result)
    
class MaxHeap:
    """
    A fully node-based Max-Heap.
    
    Operations:
    - insert(value): Adds a new node while maintaining heap order.
    - delete_max(): Removes the largest value and restores order.
    - up_heap(node): Moves a node up to maintain heap property.
    - down_heap(node): Moves a node down to maintain heap property.
    - find_last_node(): Finds the last node in level order for deletions.
    """

    def __init__(self):
        self.root = None
        self.size = 0

    def insert(self, value):
        """Insert a new value into the heap."""
        new_node = MaxHeap(value)
        if not self.root:
            self.root = new_node
            self.size = 1
            return
        
        parent = self.find_insertion_parent()
        if not parent.left:
            parent.left = new_node
        else:
            parent.right = new_node
        new_node.parent = parent
        self.size += 1

        self.up_heap(new_node)

    def delete_max(self):
        """Removes the largest element (root) and restores heap order."""
        if not self.root:
            return None
        
        max_value = self.root.value
        if self.size == 1:
            self.root = None
            self.size = 0
            return max_value

        last_node = self.find_last_node()
        self.swap_values(self.root, last_node)

        if last_node.parent:
            if last_node.parent.right == last_node:
                last_node.parent.right = None
            else:
                last_node.parent.left = None
        
        self.size -= 1
        self.down_heap(self.root)
        
        return max_value

    def up_heap(self, node):
        """Moves a node up the tree if necessary."""
        while node.parent and node.value > node.parent.value:
            self.swap_values(node, node.parent)
            node = node.parent

    def down_heap(self, node):
        """Moves a node down to maintain heap order."""
        while node.left:
            largest = node.left
            if node.right and node.right.value > node.left.value:
                largest = node.right
            
            if node.value >= largest.value:
                break
            
            self.swap_values(node, largest)
            node = largest

    def find_insertion_parent(self):
        """Finds the next available parent node for insertion using BFS."""
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            if not node.left or not node.right:
                return node
            queue.append(node.left)
            queue.append(node.right)

    def find_last_node(self):
        """Finds the last inserted node using BFS."""
        queue = [self.root]
        last = None
        while queue:
            last = queue.pop(0)
            if last.left:
                queue.append(last.left)
            if last.right:
                queue.append(last.right)
        return last

    def swap_values(self, node1, node2):
        """Swaps values of two nodes."""
        node1.value, node2.value = node2.value, node1.value

    def peek(self):
        """Returns the largest value without removing it."""
        return self.root.value if self.root else None

    def __str__(self):
        """Returns a level-order string representation of the heap."""
        result = []
        queue = [self.root]
        while queue:
            node = queue.pop(0)
            result.append(node.value)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        return str(result)


#Standard trie implementation from Ricardo:

class TrieNode():
    def __init__ (self, prefix):
        '''
        Initialisation of the Trie node
        :param prefix: The prefix is the value that the node will have itself
        :return: None
        '''
        self._children = {}
        #Dictionary with all of the children nodes from this node
        self._prefix = prefix
        #This has the node value it self
        self.complete_word = False
        #This tells us if this node is reached if the word is a complete word or not
    
    def _add_word(self, word):
        '''
        Given a word, it will add it to the trie recursively
        :param word: The word that will be added to the trie
        :return: None
        '''
        
        if len(word) == 0:
            #Base case when the word has been completely added
            self.complete_word = True
            return 

        #Checks if the first letter is not a note yet
        if word[0] not in self._children:
            #If its not there, It creates a child with that letter
            self._children[str(word[0])] = TrieNode(word[0])
        
        #Then it continues creating that word in the dictionary
        self._children[str(word[0])]._add_word(word[1:])

    def _find_word(self,word):
        '''
        Given a word, it will traverse the node and find out of that word has been added
        to it before or not
        :param word: The word that will be found to the trie
        :return: True if its a complete word in the tree, false if not
        '''
    
        if len(word) == 0:
            #Base case when we have reached the end of the word
            return self.complete_word
        
        if word[0] in self._children:
            #Traversing the node if the next letter from the word is found
            return self._children[str(word[0])].find_word(word[1:])
        else:
            #Retuns false because it hasnt been found
            return False

    def _output_tree(self, level=0):
        '''
        Outputs the tree
        '''
        for child in self._children:
            print("|"+"-"*level + child)
            self._children[child]._output_tree(level+1)

#If someone feels inspired they can fill in compact trie, compressed trie and suffix trie

#Grahs:
'''
What is a graph?
A graph is a data strucutre that has nodes and edges. The nodes are connected to each other by edges.
    -Graphs have no roots
    -Graphs can be connected or not
        - A connected graph is a graph whose all nodes are connected by edges.
    -Two edges are called parallel when they are incident with the same nodes.
    -An edge is called a loop when it connects a node with itself. 
    - Graphs can be simple or complex
        -Simple means that it has no loops nor parallel edges
        -Complex is the rest
    - A path is a series of edges that are possible to take to go from one node to another.
    - A graph can also be weighted. This means that the edges have a value/weight/cost attributed to them.

    -There is two ways to search in a graph:
        -Depth first search - You travel as far as you can into the graph, when you find a dead end, you back track to the previous moment where 
            you could have taken another route and continue from there.
        -Breath first search - Systematically go from nodes that are one edge away from the begining to two...
        Both have a time complexity of O ( m + n ) when using an adjacency list and O(N^2) for when using an adjancency matrix.
    
    - There is two algorithms that we have to know to find the shortest path between two nodes in a graph:
        -Dijkstra's algorithm - Essentially, you visit every node and keep track of the shortest distance from the start to that node
            -The next node you visit is always the one with the smallest value, guarranteeing that the distance up to that point is the smallest it can be
            -Repeat untill there is no nodes left and you have the value of the shortest path
        -A* algorithm: Similar to Dijkstra's algorithm, but instead of only considering the shortest distance from the start, it also takes into account an estimated distance to the goal.
            -The next node you visit is the one with the smallest sum of the known distance from the start and the estimated remaining distance (heuristic), ensuring that paths leading towards the goal are prioritized.
            -Repeat until you reach the goal, guaranteeing the shortest path while searching more efficiently than Dijkstra's. 
'''

#Ricardo's implementation of a graph (UNWEIGHTED)

class GraphNode():
    def __init__(self, identifier):
        '''
        Initialises a node in the graph
        '''
        self._identifier = identifier
        self._connected = []

    def add_connected(self,node):
        '''
        This funciton adds a node connected to GraphNode
        '''
        #Since connections are both ways, we also have to add the current node as a child to the other node
        if node not in self._connected:
            self._connected.append(node)
            node.add_connected(self)

#Weighted:
class GraphNode():
    def __init__(self, identifier):
        '''
        Initialises a node in the graph
        '''
        self._identifier = identifier
        self._connected = {} #Dictionary of the nodes:weight

    def add_connected(self,node,weight):
        '''
        This funciton adds a node connected to GraphNode
        '''
        #Since connections are both ways, we also have to add the current node as a child to the other node
        if node not in self._connected:
            self._connected[node] = weight
            node.add_connected(self,weight)

#Harmen's class implementation for weighted graphs
class GraphEdge:
    def __init__(self, origin: int, destination: int, weight: float = 1.0):
        self._origin = origin
        self._destination = destination
        self._weight = weight

    def is_incident(self, node: int) -> bool:
        return node == self._origin or node == self._destination

    def other_node(self, node: int) -> int:
        if self.is_incident(node):
            return self._origin + self._destination - node
        return -1

    def get_weight(self) -> float:
        return self._weight

class UndirectedGraph:
    def __init__(self, node_count: int):
        self._neighbours = [[] for _ in range(node_count)]

    def add_edge(self, node1: int, node2: int, weight: int = 1):
        new_edge = GraphEdge(node1, node2, weight)
        self._neighbours[node1].append(new_edge)
        self._neighbours[node2].append(new_edge)

#Searches in a graph:

#DFS and BFS harmen version (They have to be inside of his class):
def depth_first_search(self, start_node: int, action, visited: list[bool] = None) -> list[bool]:
    if visited is None:
        visited = [False] * len(self._neighbours)

    action(start_node)
    visited[start_node] = True

    for edge in self._neighbours[start_node]:
        other_node = edge.other_node(start_node)
        if not visited[other_node]:
            self.depth_first_search(other_node, action, visited)

    return visited

def breadth_first_search(self, start_node: int, action) -> list[bool]:
    visited = [False] * len(self._neighbours)
    q = Queue()
    
    visited[start_node] = True
    q.enqueue(start_node)

    while q.size() > 0:
        current_node = q.dequeue()
        action(current_node)

        for edge in self._neighbours[current_node]:
            other_node = edge.other_node(current_node)
            if not visited[other_node]:
                visited[other_node] = True
                q.enqueue(other_node)

    return visited

#DFS Ricardo (Has to be inside of the ricardo node class):
def dfs(self, visited = [], connected = set()):
    visited.append(self)
    connected.add(self._value)
    
    for child in self._children:
        if child not in visited:
            connected.add(child)
            #Do whatever you want to do with the visited node here
            child.dfs(visited, connected)
    
    return connected

'''
Why no negative values in dijktras? (CHAT GPT)
Dijkstra's algorithm doesn't work correctly with negative edge weights because it assumes that once a node's shortest distance is found, it will not change. This assumption is broken when negative weights are introduced. Here's why:
    - Priority Queue Assumption
    - Dijkstra's algorithm picks the node with the smallest known distance and assumes that all shorter paths to other nodes have already been considered.
    - If an edge with a negative weight is later encountered, it could provide a shorter path to a node that was already finalized, but the algorithm doesn't go back and update it.'''

#By Chirag:
#Dijkstra Algorithm with priority queue = (O((V + E) log V))

"""
Shortest Path
Steps: 
    Assign all nodes a distance of infinity (∞), except the source node (0).
    Use a priority queue (min-heap) to select the node with the smallest known distance.
    Update distances for its neighbors if a shorter path is found.
    Repeat until all nodes have been visited.
    
Implementation: 
import heapq

def dijkstra(graph, start):
    pq = [(0, start)]  # Min-heap (priority queue)
    distances = {node: float('inf') for node in graph}
    distances[start] = 0

    while pq:
        current_distance, node = heapq.heappop(pq)

        for neighbor, weight in graph[node]:
            distance = current_distance + weight

            if distance < distances[neighbor]:  # Found a shorter path
                distances[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))

    return distances
#example: 
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('A', 1), ('C', 2), ('D', 5)],
    'C': [('A', 4), ('B', 2), ('D', 1)],
    'D': [('B', 5), ('C', 1)]
}

print("\nShortest Distances from A:", dijkstra(graph, 'A'))

"""

#A* Algorithm: is dijkstra but uses heuristics = (O((V + E) log V))
"""
    It balances:
        g(n): Cost from the start node to n
        h(n): Estimated cost from n to the goal
        f(n) = g(n) + h(n)
        
    Steps of A* Algorithm
        Start from the initial node.
        Use a priority queue, prioritizing nodes with the lowest f(n).
        Update g(n) and f(n).
        Stop when the goal is reached.
        
Python Implementation:
    import heapq

def a_star(graph, start, goal, heuristic):
    pq = [(0, start)]  # Min-heap
    g_costs = {node: float('inf') for node in graph}
    g_costs[start] = 0
    came_from = {}

    while pq:
        _, node = heapq.heappop(pq)

        if node == goal:
            break

        for neighbor, weight in graph[node]:
            tentative_g = g_costs[node] + weight

            if tentative_g < g_costs[neighbor]:
                g_costs[neighbor] = tentative_g
                f_score = tentative_g + heuristic[neighbor]
                heapq.heappush(pq, (f_score, neighbor))
                came_from[neighbor] = node

    return g_costs

heuristic = {'A': 4, 'B': 2, 'C': 1, 'D': 0}
print("\nA* Shortest Path from A to D:", a_star(graph, 'A', 'D', heuristic))

"""
# Some commen exercises (they were taken from leetcode)
# Arrays
class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        A, B = len(word1), len(word2) # in this excersis you have to merge two sentences into one --> abc + def = adbecf 
        a, b = 0, 0   # Let A be the length of Word1  # Let B be the length of Word2
        s = []                # Let T = A + B  # Time: O(T) # Space: O(T)
        word = 1
        while a < A and b < B:
            if word == 1:
                s.append(word1[a])
                a += 1
                word = 2
            else:
                s.append(word2[b])
                b += 1
                word = 1
        while a < A:
            s.append(word1[a])
            a += 1
        while b < B:
            s.append(word2[b])
            b += 1
        return ''.join(s)

class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        S = len(s)    # Time: O(T)  # Space: O(1) 
        T = len(t)    # check if str[1] is sub of str[2] --> abc in anmbfghd # True
        if s == '': return True
        if S > T: return False
        j = 0
        for i in range(T):
            if t[i] == s[j]:
                if j == S-1:
                    return True
                j += 1
        return False
#Linked Lists

class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur = head
        while cur and cur.next:           # Time Complexity: O(n) # Space Complexity: O(1)
            if cur.val == cur.next.val:    #Remove Duplicates from Sorted List
                cur.next = cur.next.next
            else:
                cur = cur.next
        return head

class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        d = ListNode()
        cur = d
        while list1 and list2:         # Time Complexity: O(n)  # Space Complexity: O(1)
            if list1.val < list2.val:  #(1)-->(1)-->(4) + (2)-->(3) = (1)-->(1)-->(2)-->(3)-->(4)
                cur.next = list1
                cur = list1
                list1 = list1.next
            else:
                cur.next = list2
                cur = list2
                list2 = list2.next

        cur.next = list1 if list1 else list2
        return d.next

class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur = head           # Time Complexity: O(n) # Space Complexity: O(1)
        prev = None           # Reverse a list
        while cur:
            temp = cur.next
            cur.next = prev
            prev = cur
            cur = temp
        return prev

# Binary search 
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left = 0
        right = len(nums) - 1                # Time Complexity: O(log(n)) # Space Complexity: O(1)
        while left <= right:
            middle = (right + left) // 2
            if nums[middle] == target:
                return middle
            elif nums[middle] > target:
                right = middle - 1
            else:
                left = middle + 1
        return -1

class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        n = len(nums)              #Given a sorted array of distinct integers and a target value,
        l = 0     #return the index if the target is found. If not, return the index where it would be.
        r = n - 1                 # Time Complexity: O(log n) # Space Complexity: O(1)
        while l <= r:
            m = (l + r) // 2
            if nums[m] < target:
                l = m + 1
            elif nums[m] > target:
                r = m - 1
            else:
                return m
        if nums[m] < target:
            return m + 1
        else:
            return m
# Trees
class Solution:
    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:   # Time Complexity: O(n) # Space Complexity: O(h) { here "h" is the height of the tree }
            return None
        root.left, root.right = root.right, root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        return root
# DFS Max depth
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        if not root: # Time Complexity: O(n)
            return 0 # Space Complexity: O(h) { here "h" is the height of the binary tree }
        left = self.maxDepth(root.left)
        right = self.maxDepth(root.right)
        return 1 + max(left, right)
# BFS Max depth
class Solution:
    def maxDepth(self, root):
        if not root:
            return 0  # Height of an empty tree is 0
        queue = deque([root])
        height = 0
        while queue:
            level_size = len(queue)  # Number of nodes at the current level
            for _ in range(level_size):
                node = queue.popleft()      # Time Complexity: O(n) # Space Complexity: O(n)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)     
            height += 1  # Increment height at each level
        return height

#Recursive Backtracking
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)        # example: Input: nums = [1,2,3]  #Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
        ans, sol = [], []    # Time Complexity: O(2^n)
        def backtrack(i):    # Space Complexity: O(n)
            if i == n:
                ans.append(sol[:])
                return  # Don't pick nums[i]
            backtrack(i + 1)  # Pick nums[i]
            sol.append(nums[i])
            backtrack(i + 1)
            sol.pop()
        backtrack(0)
        return ans

#Graphs
#here is a bi-directional graph with n vertices, where each vertex is labeled from 0 to n - 1 (inclusive).
The edges in the graph are represented as a 2D integer array edges, where each edges[i] = [ui, vi] denotes a bi-directional edge
between vertex ui and vertex vi. Every vertex pair is connected by at most one edge, and no vertex has an edge to itself.
You want to determine if there is a valid path that exists from vertex source to vertex destination.Given edges and the integers n,
source, and destination, return true if there is a valid path from source to destination, or false otherwise.
#there are 3 diffrents approches 

# Recursive DFS
class Solution:
    def validPath(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        if source == destination:
            return True
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        seen = set()
        seen.add(source)
        def dfs(i):
            if i == destination:
                return True
            for nei_node in graph[i]:
                if nei_node not in seen:
                    seen.add(nei_node)
                    if dfs(nei_node):
                        return True
            return False  
        return dfs(source)
# Iterative DFS with Stack
class Solution:
    def validPath(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        if source == destination:
            return True
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        seen = set()
        seen.add(source)
        stack = [source]
        while stack:
            node = stack.pop()
            if node == destination:
                return True
            for nei_node in graph[node]:
                if nei_node not in seen:
                    seen.add(nei_node)
                    stack.append(nei_node)
        return False
# BFS With Queue
from collections import deque
class Solution:
    def validPath(self, n: int, edges: List[List[int]], source: int, destination: int) -> bool:
        if source == destination:
            return True
        graph = defaultdict(list)
        for u, v in edges:
            graph[u].append(v)
            graph[v].append(u)
        seen = set()
        seen.add(source)
        q = deque()
        q.append(source)
        while q:
            node = q.popleft()
            if node == destination:
                return True
            for nei_node in graph[node]:
                if nei_node not in seen:
                    seen.add(nei_node)
                    q.append(nei_node)
        return False # Time: O(N + E), Space: O(N + E)

"""
Tutorial Code / Additional Code
"""

#2.3: Stack from one or more queues

class StackUsingQueue:
    def __init__(self):
        self.queue1 = Queue()
        self.queue2 = Queue()

    def push(self, item):
        self.queue1.enqueue(item)

    def pop(self):
        if self.queue1.size() == 0:
            print("Stack is empty!")
            return None

        # Move all elements from queue1 to queue2, except the last one
        while self.queue1.size() > 1:
            self.queue2.enqueue(self.queue1.dequeue())
        
        # The last element in queue1 is the one to pop
        item = self.queue1.dequeue()

        # Swap queues
        self.queue1, self.queue2 = self.queue2, self.queue1

        return item
    
#2.4: Checking if parenthetical expression in balanced

def is_balanced(s):
    stack = Stack()
    for i in s:
        if i == "(" or i == "[":
            stack.push(i)
        elif i == "]":
            if stack.size() == 0 or stack.pop() != "[":
                print("unbalanced")
                return
        elif i == ")":
            if stack.size() == 0 or stack.pop() != "(":
                print("unbalanced")
                return
    if stack.size() != 0:
        print("unbalanced")
    else:
        print("balanced")

#2.6: Pop method in Queue()

def pop(self):
    """
    Retrieve and remove the value that was added last
    :return: the item that was added last to the queue
    """
    if self.size() == 0:
        print("Queue empty!")
        return None
    
    val = self._q[self._back - 1]
    self._back = (self._back - 1) % len(self._q)
    return val

#2.7: Pop in stack removes max element

def pop(self):
    """
    Removes and returns the item on top of the stack
    :return: item on top of the stack
    """
    max_ = max(self._stack)
    self._stack.remove(max_)
    return max_

#2.8: Copy a stack

def copy_stack_recursively(stack):
    """
    Function to copy a stack recursively.
    """
    # Base case: If the stack is empty, return an empty list (representing an empty stack)
    if not stack:
        return []

    # Step 1: Pop the top element from the stack
    top_element = stack.pop()

    # Step 2: Copy the rest of the stack recursively
    copied_stack = copy_stack_recursively(stack)

    # Step 3: Push the popped element to the copied stack to maintain order
    copied_stack.append(top_element)

    # Step 4: Push the element back to the original stack to preserve its state
    stack.append(top_element)

    return copied_stack

#2.11: LinkedList method to find whether a particular value exists. (Iterative solution)

def find(self, value):
    while self._next is not None and self._next._item != value:
        self = self._next
    if self._next is None:
        return False
    else:
        return True

#2.12: LinkedList method to remove all nodes from a list with a given value
def remove_all(self, value):
    if self._next is None:
        return
    if self._next._item == value:
        self.remove_first_item()
        self.remove_all(value)
    else:
        self._next.remove_all(value)

#2.13: LinkedList method to remove a node at a given position from a list
def remove_at_index(self, pos):
    if pos > self.size():
        return self._list_too_short_error()
    if pos == 0:
        self.remove_first_item()
    if self._next is not None:
        return self._next.remove_at_index(pos - 1)
    
#2.15 LinkedList method to remove last occurence of a given value
def remove_last_occurrence(self, value) -> None:

    prev_last = None  # Track predecessor of the last occurrence
    prev = self       # Current predecessor node during traversal
    current = self._next  # Current node being checked
    
    # Traverse the entire list
    while current is not None:
        if current._item == value:
            prev_last = prev  # Update last found position
        prev = current
        current = current._next
    
    # Remove the node if found
    if prev_last is not None:
        prev_last._next = prev_last._next._next
    else:
        self._value_not_found_error()

#2.16: LinkedList method to remove consecutive duplicate values
def remove_duplicates(self):
    if self._next is None:
        return
    if self._next._next is not None and self._next._item == self._next._next._item:
        self._next = self._next._next  # Remove duplicate
        self.remove_duplicates()  # Continue checking
    else:
        self._next.remove_duplicates()  # Move to next node


#3.3 a) Function to copy a tree
def copy_tree(tree: TreeNode) -> TreeNode:
    if tree is None:
        return None
    return TreeNode(tree._item, copy_tree(tree._left), copy_tree(tree._right))

#3.3 b) Get height of tree
def tree_height(tree: TreeNode) -> int:
    if tree is None:
        return -1  # Return -1 so that a single-node tree has height 0
    
    left_height = tree_height(tree._left)
    right_height = tree_height(tree._right)
    
    return 1 + max(left_height, right_height)


#3.3 c) Number of nodes in tree
def node_number(tree: TreeNode) -> int:
    if tree is None:
        return 0
    return 1 + node_number(tree._left) + node_number(tree._right)

#3.3 d) Find node with max value
def find_max_node(tree: TreeNode) -> int | None:
    if not tree:
        return None
    current = tree
    while current._right is not None:
        current = current._right
    return current._item

#3.3 e) Highest position that contains a node

def is_search_tree(root: TreeNode) -> bool:
    def validate(node, low=-float('inf'), high=float('inf')):
        if not node:
            return True
        if not (low < node._item < high):
            return False
        return (validate(node._left, low, node._item) and 
                validate(node._right, node._item, high))
    return validate(root)

#3.4: Finds node with a given position, n

def find_node(tree: TreeNode, pos: int) -> TreeNode:
    if tree is None:
        return None
    
    if pos == 1:
        return tree
    
    parent_pos = pos // 2

    if pos % 2 == 0:
        return find_node(tree._left, parent_pos)
    else:
        return find_node(tree._right, parent_pos)
    
#3.7 Tree to list
def tree_to_list(tree: TreeNode, index=1, result=None) -> list:
    if result is None:
        result = []
    
    # Ensure the list is large enough for the current index
    while len(result) < index:
        result.append(None)
    
    result[index - 1] = tree._item  # 1-based index adjustment

    if tree._left:
        tree_to_list(tree._left, index * 2, result)  # Left child at index * 2
    if tree._right:
        tree_to_list(tree._right, index * 2 + 1, result)  # Right child at index * 2 + 1

    return result

#3.8 a) Inorder Successor
def inorder_successor(node: TreeNode) -> TreeNode:
    if node._right is not None:
        w = node._right
        while w._left is not None:
            w = w._left
        return w
    return None


    
def remove_from_search_tree(tree: TreeNode, value: int) -> TreeNode:
    # Step 1: Search for the node to remove
    node = tree
    parent = None
    
    # Find the node
    while node is not None and node._item != value:
        parent = node
        if value < node._item:
            node = node._left
        else:
            node = node._right

    if node is None:
        return tree  # Value not found, return tree as is

    # Step 2: Node found, remove the node based on number of children
    # Case 1: Node has no children (leaf node)
    if node._left is None and node._right is None:
        if parent is None:
            return None  # If root node is to be deleted, return None
        if parent._left == node:
            parent._left = None
        else:
            parent._right = None
    
    # Case 2: Node has one child
    elif node._left is None or node._right is None:
        child = node._left if node._left is not None else node._right
        if parent is None:
            return child  # If root node is to be deleted, return its only child
        if parent._left == node:
            parent._left = child
        else:
            parent._right = child
    
    # Case 3: Node has two children
    else:
        # Find the inorder successor
        successor = inorder_successor(node)
        
        # Replace node's value with the successor's value
        node._item = successor._item
        
        # Now remove the successor node
        remove_from_search_tree(node._right, successor._item)  # Successor is always in the right subtree

    return tree

#3.10: Function to upheap (already there)

#3.11: Function to downheap (already there)


