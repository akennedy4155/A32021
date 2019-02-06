# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    # make the root        
    root = util.TreeNode((problem.getStartState(), None, 0), problem.getStartState())

    # start fringe with root node
    fringe = util.Stack()
    fringe.push(root)
    visited = set()

    while True:
        if fringe.isEmpty():
            return "failure"

        # get the current node and the current pos
        currentNode = fringe.pop()
        currentPos = currentNode.state

        # check if the node has already been expanded
        if currentPos not in visited:
            # add that node to visited
            visited.add(currentPos)

            # if the problem is a goal state, return the path that it found to get there
            if problem.isGoalState(currentPos):
                return currentNode.getPathToRoot()

            # else keep expanding according to the fringe
            else:
                # expand the node and add it to the tree
                for connex in problem.getSuccessors(currentPos):
                    newNode = util.TreeNode(connex, connex[0], parent = currentNode)
                    fringe.push(newNode)

# https://stackoverflow.com/questions/1649027/how-do-i-print-out-a-tree-structure
# translated from c# to python
def printTree(tree, indent, last):
    print(indent + "+-" + str(tree.value))
    if last:
        indent += "   "
    else:
        indent += "|  "

    for i in range(len(tree.children)):
        printTree(tree.children[i], indent, i == len(tree.children) - 1)

def breadthFirstSearch(problem):
    # make the root        
    root = util.TreeNode((problem.getStartState(), None, 0), problem.getStartState())

    # start fringe with root node
    fringe = util.Queue()
    fringe.push(root)
    visited = set()

    while True:
        if fringe.isEmpty():
            return "failure"

        # get the current node and the current pos
        currentNode = fringe.pop()
        currentState = currentNode.state

        # check if the node has already been expanded
        if currentState not in visited:
            # add that node to visited
            visited.add(currentState)

            # if the problem is a goal state, return the path that it found to get there
            if problem.isGoalState(currentState):
                return currentNode.getPathToRoot()

            # else keep expanding according to the fringe
            else:
                # expand the node and add it to the tree
                for connex in problem.getSuccessors(currentState):
                    newNode = util.TreeNode(connex, connex[0], parent = currentNode)
                    fringe.push(newNode)

def uniformCostSearch(problem):
    # make the root        
    root = util.TreeNode((problem.getStartState(), None, 0), problem.getStartState())

    # start fringe with root node
    fringe = util.PriorityQueue()
    fringe.push(root, problem.getCostOfActions(root.getPathToRoot()))
    visited = set()

    while True:
        if fringe.isEmpty():
            return "failure"

        # get the current node and the current pos
        currentNode = fringe.pop()
        currentPos = currentNode.state

        # check if the node has already been expanded
        if currentPos not in visited:
            # add that node to visited
            visited.add(currentPos)

            # if the problem is a goal state, return the path that it found to get there
            if problem.isGoalState(currentPos):
                return currentNode.getPathToRoot()

            # else keep expanding according to the fringe
            else:
                # expand the node and add it to the tree
                for connex in problem.getSuccessors(currentPos):
                    newNode = util.TreeNode(connex, connex[0], parent = currentNode)
                    fringe.push(newNode, problem.getCostOfActions(newNode.getPathToRoot()))

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
        # make the root        
    root = util.TreeNode((problem.getStartState(), None, 0), problem.getStartState())

    # start fringe with root node
    fringe = util.PriorityQueue()
    aStarVal = problem.getCostOfActions(root.getPathToRoot()) + heuristic(problem.getStartState(), problem)
    fringe.push(root, aStarVal)
    visited = set()

    while True:

        # get the current node and the current pos
        currentNode = fringe.pop()
        currentPos = currentNode.value[0]

        # check if the node has already been expanded
        if currentPos not in visited:
            # add that node to visited
            visited.add(currentPos)

            # if the problem is a goal state, return the path that it found to get there
            if problem.isGoalState(currentPos):
                return currentNode.getPathToRoot()

            # else keep expanding according to the fringe
            else:
                # expand the node and add it to the tree
                for connex in problem.getSuccessors(currentPos):
                    newNode = util.TreeNode(connex, connex[0], parent = currentNode)
                    aStarVal = problem.getCostOfActions(newNode.getPathToRoot()) + heuristic(newNode.value[0], problem)
                    fringe.update(newNode, aStarVal)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
