# ads-sheet



#Trees 
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
