# ads-sheet



#Trees 
    #Definitions
        #Binary Tree
            # Def: Tree where each node has ≤ 2 children
            # Height h → max 2^h leaves
            # in binary tree with n nodes, height can vary from O(n) to O(log(n))
        #Perfect Binary Tree
            # Def: All levels fully filled, all leaves at same depth
            # Nodes = 2^(h+1) - 1 (where h = height)
        #Complete Binary Tree
            # Def: All levels except last fully filled, last level maximally left-aligned
        #Heap
            # Def: Complete binary tree with heap property
            # -Min-Heap: Parent ≤ children.
            # -Max-Heap: Parent ≥ children
        #Binary search tree
            #Def: Binary tree with left < parent < right ordering
Code Section:
    # Preorder Traversal (Root, Left, Right)
    def preorder(root):
        if root:
            current_val = root.value
            preorder(root.left)
            preorder(root.right)

    # Inorder Traversal (Left, Root, Right)
    def inorder(root):
        if root:
            inorder(root.left)
            current_val = root.value
            inorder(root.right)

    # Postorder Traversal (Left, Right, Root)
    def postorder(root):
        if root:
            postorder(root.left)
            postorder(root.right)
            current_val = root.value
Pseudocode Section:
