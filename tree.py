from copy import copy, deepcopy
import operator
import pprint
import random
from typing import Union
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
import matplotlib.pyplot as plt
import numpy as np
import inspect

def normalize(vector: list[float]) -> list[float]:
    vector: np.ndarray = np.array(vector)
    
    if np.sum(vector) == 1:
        return vector.tolist()
    
    return (vector / np.sum(vector)).tolist()

class Operator:
    def __init__(self, symbol, function, in_params, weight):
        self.symbol = symbol
        self.function = function
        self.fanin = in_params
        self.weight = weight
        
    def __call__(self, *args):
        return self.function(*args)

    def __str__(self):
        return self.symbol

    def __repr__(self):
        return self.symbol

class Tree:
    def __init__(self, max_depth, operators: list[Operator], variables, constants):
        self.max_depth = max_depth
        self.operators: list[Operator] = operators
        self.variables = variables
        self.constants = constants
        self.root = None
        self.children = []
        
    @property
    def is_leaf(self):
        return not self.children
        
    def _create_random_subtree(self, depth):
        if depth >= self.max_depth or random.random() < depth / self.max_depth:
            # Create a leaf node (variable or constant)
            return Tree(self.max_depth, self.operators, self.variables, self.constants), random.choice(self.variables + self.constants)
        else:
            # Create an operator node
            operator: Operator = random.choice(self.operators)
            num_params = operator.fanin
            tree = Tree(self.max_depth, self.operators, self.variables, self.constants)
            tree.root = operator
            for _ in range(num_params):
                child, child_value = self._create_random_subtree(depth + 1)
                child.root = child_value
                tree.children.append(child)
                
                assert child.root is not None
            return tree, operator

    @staticmethod
    def create_individual(operators, variables, constants, max_depth=10):
        """Create a random individual (tree) with a maximum depth of max_depth."""
        tree = Tree(max_depth, operators, variables, constants)
        tree, _ = tree._create_random_subtree(0)
        return tree

    def mutate_point(self):
        nodes = self.get_all_nodes()
        node, _ = random.choice(nodes)
        if not node.children:
            # It's a leaf node
            if random.random() < 0.8:
                node.root = random.choice(self.variables + self.constants)
            else:
                node.root = random.choice(self.constants)
        else:
            # It's an operator node
            node.root = random.choice([operator for operator in self.operators if operator.fanin == node.root.fanin])
            
        return node

    def mutate_subtree(self):
        """Mutate a random subtree with another randomly generated subtree."""

        nodes = self.get_all_nodes()
        subtree, current_depth = random.choice(nodes)
        num_children = len(subtree.children)
        
        if num_children > 1:
            subtree.children[random.randint(0, num_children-1)], _ = self._create_random_subtree(current_depth+1)
        elif num_children == 1:
            subtree.children[0], _ = self._create_random_subtree(current_depth+1)

    def mutate_permute(self):
        """Permute the operands of a randomly chosen subtree"""

        nodes = self.get_all_nodes()
        node, _ = random.choice(nodes)
        random.shuffle(node.children)

    def mutate_hoist(self):
        """Replace the entire tree with a randomly chosen subtree."""
        nodes = self.get_all_nodes()
        if len(nodes) > 1:
            subtree, _ = random.choice(nodes[1:])
            self.root = subtree.root
            self.children = subtree.children
            
        assert self.get_all_nodes()

    def mutate_expand(self):
        """Expand a randomly chosen leaf into an operator with operands."""

        leaves = self.get_all_leaves()
        if leaves:
            leaf = random.choice(leaves)
            new_subtree, _ = self._create_random_subtree(0)
            leaf.root = new_subtree.root
            leaf.children = new_subtree.children

    def mutate_collapse(self):
        """Collapse a randomly chosen subtree into a leaf."""
        nodes = self.get_all_nodes()
        if len(nodes) > 1:
            subtree, _ = random.choice(nodes[1:])
            leaf_value = random.choice(self.variables + self.constants)
            subtree.root = leaf_value
            subtree.children = []
            
    @staticmethod
    def mutate(tree: "Tree") -> "Tree":
        """Perform a random mutation on the tree with a given probability."""
        new_tree: Tree = deepcopy(tree)
        # mutation = random.choice([
        #     new_tree.mutate_point,
        #     #new_tree.mutate_subtree,
        #     # new_tree.mutate_permute,
        #     # new_tree.mutate_hoist,
        #     # new_tree.mutate_expand,
        #     # new_tree.mutate_collapse
        # ])
        
        new_tree.mutate_point()
        
        return new_tree

    @staticmethod
    def crossover(tree1: "Tree", tree2: "Tree") -> "Tree":
        """Perform crossover between two trees by exchanging a random subtree from tree1 with a subtree from tree2."""
        new_tree = deepcopy(tree1)
        nodes_tree1 = new_tree.get_all_nodes()
        nodes_tree2 = tree2.get_all_nodes()

        if len(nodes_tree1) > 1 and len(nodes_tree2) > 1:
            subtree1, _ = random.choice(nodes_tree1[1:])
            subtree2, _ = random.choice(nodes_tree2[1:])
            subtree1.root = subtree2.root
            subtree1.children = deepcopy(subtree2.children)
            
            #assert subtree1.root.fanin == len(subtree1.children)

        return new_tree
            
    def draw(self):
        """Draw the tree using networkx in a tree-like style."""
        G = nx.DiGraph()
        labels = {}
        node_id = 0

        def add_nodes_edges(node, parent=None):
            nonlocal node_id
            current_id = node_id
            node_id += 1
            if parent is not None:
                G.add_edge(parent, current_id)
            labels[current_id] = str(node.root.symbol) if node.children else (str(f"{node.root:.3f}") if isinstance(node.root, float) else str(node.root))
            for child in node.children:
                add_nodes_edges(child, current_id)
                
        def hierarchy_pos(G: nx.DiGraph, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
            """
            If there is a cycle, then this will produce a hierarchy, but nodes will be repeated.
                G: the graph (must be a tree)
                root: the root node of current branch
                width: horizontal space allocated for this branch - avoids overlap with other branches
                vert_gap: gap between levels of hierarchy
                vert_loc: vertical location of root
                xcenter: horizontal location of root
            """
            pos = {root: (xcenter, vert_loc)}
            neighbors = list(G.neighbors(root))
            if len(neighbors) != 0:
                dx = width / len(neighbors) 
                nextx = xcenter - width/2 - dx/2
                for neighbor in neighbors:
                    nextx += dx
                    pos.update(hierarchy_pos(G, neighbor, width=dx, vert_gap=vert_gap, vert_loc=vert_loc-vert_gap, xcenter=nextx))
            return pos

        plt.figure(figsize=(14,8))
        
        add_nodes_edges(self)
        
        if not G.nodes:
            G.add_node(0)
        
        pos = hierarchy_pos(G, min(G.nodes))
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=400, node_color='lightblue', font_size=10)

    def get_all_nodes(self):
        """Helper method to get all nodes in the tree."""
        nodes = []
        stack = [(self, 0)]

        while stack:
            node, depth = stack.pop()
            nodes.append((node, depth))
            children = list(zip(node.children, [depth + 1] * len(node.children)))
            stack.extend(children)
            
        return nodes
    
    def get_all_leaves(self):
        """Helper method to get all leaves in the tree."""
        leaves = []

        def traverse(node):
            if not node.children:
                leaves.append(node)
            for child in node.children:
                traverse(child)

        traverse(self)
        return leaves
    
    @staticmethod
    def evaluate(tree: "Tree", X: np.array) -> np.array:
        """Evaluate the tree for a given input."""
        def evaluate_node(node, x):
            
            if not node.children:
                
                # It's a leaf node
                if node.root in tree.variables:
                    return x[tree.variables.index(node.root)]
                else:
                    return node.root
            else:
                # It's an operator node
                params = [evaluate_node(child, x) for child in node.children]
                
                return node.root(*params)

        #print(X.T.shape, tree.variables)
        return np.array([evaluate_node(tree, row) for row in X.T])
    
    @staticmethod
    def mse(tree: "Tree", X: np.array, y_target: np.array) -> float:
        y_pred = Tree.evaluate(tree, X)
        return (sum((a - b) ** 2 for a, b in zip(y_target, y_pred)) / len(y_target)).astype(float)

    def __repr__(self):
        """Retrieve the formula from the tree by performing an in-order traversal."""
        def traverse(node):
            if not node.children:
                return f"{node.root:.3f}" if isinstance(node.root, float) else str(node.root)
            left_expr = traverse(node.children[0])
            right_expr = traverse(node.children[1]) if len(node.children) > 1 else ""
            return f"({left_expr} {node.root.symbol} {right_expr})"

        return traverse(self)

if __name__ == "__main__":
    
    import sys
    w = [1, 2, 10]
    r = normalize(w)
    print(w, r)
    sys.exit()
    
    # Define some example operators and variables
    built_in_operators = [
        Operator('+', operator.add, 2),
        Operator('-', operator.sub, 2),
        Operator('*', operator.mul, 2),
        Operator('/', operator.truediv, 2),
        Operator('^', operator.pow, 2),
        Operator('|.|', operator.abs, 1),
        Operator('-', operator.neg, 1)
    ]
    
    numpy_operators = [
        Operator("|.|", np.abs, 1),
        Operator("-", np.negative, 1),
        Operator("+", np.add, 2),
        Operator("-", np.subtract, 2),
        Operator("*", np.multiply, 2),
        Operator("/", np.divide, 2),
        Operator("^", np.power, 2),
        Operator("sign", np.sign, 1),
        Operator("exp", np.exp, 1),
        Operator("exp2", np.exp2, 1),
        Operator("sqrt", np.sqrt, 1),
        Operator("square", np.square, 1),
        Operator("cbrt", np.cbrt, 1),
        Operator("reciprocal", np.reciprocal, 1),
        Operator("sin", np.sin, 1),
        Operator("cos", np.cos, 1),
        Operator("tan", np.tan, 1),
        Operator("sinh", np.sinh, 1),
        Operator("cosh", np.cosh, 1),
        Operator("tanh", np.tanh, 1)
    ]
    
    critical_operators = [
        Operator("log", np.log, 1),
        Operator("log2", np.log2, 1),
        Operator("log10", np.log10, 1),
        Operator("arcsin", np.arcsin, 1),
        Operator("arccos", np.arccos, 1),
        Operator("arctan", np.arctan, 1)
    ]
    
    operators = built_in_operators + numpy_operators
    
    variables = ['x0', 'x1']
    constants = [10 * random.random() for _ in range(5)]
    data = np.load("./data/problem_0.npz")
    
    for i in range(100):
        individual = Tree.create_individual(max_depth=5, operators=operators, variables=variables, constants=constants)
        print(f"Original Tree: {individual}")
        individual.draw()
        
        X = np.random.rand(2,3)
        y = np.random.randint(0, 10, 3)
        
        print(f"Shapes: {X.shape}, {y.shape}")
        
        print(Tree.mse(individual, X, y))

        individual = Tree.mutate(individual)
        print(f"After Point Mutation: {individual}")
        individual.draw()

        # individual.mutate_subtree()
        # print("After Subtree Mutation:")
        # individual.draw()

        # individual.mutate_permute()
        # print("After Permute Mutation:")
        # individual.draw()

        # individual.mutate_hoist()
        # print("After Hoist Mutation:")
        # individual.draw()

        # individual.mutate_expand()
        # print("After Expand Mutation:")
        # individual.draw()

        # individual.mutate_collapse()
        # print("After Collapse Mutation:")
        # individual.draw()

        # other_individual = Tree.create_individual(max_depth=10, operators=operators, variables=variables, constants=constants)
        # print("Other Individual Tree:")
        # other_individual.draw()

        # Tree.exchange_subtrees(individual, other_individual)
        # print("After Subtree Exchange:")
        # individual.draw()
        # other_individual.draw()
        
        #plt.show()