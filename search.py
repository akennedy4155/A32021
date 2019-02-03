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

def depthFirstSearchNaiveStart(problem):
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
    # initialize the search tree of the problem

    g = Graph(map = problem)
    # build a tree with root node of the start state formatted into a tuple
    # like the one that the successor function returns
    currentNode = util.TreeNode(successorState = (problem.getStartState(), None, 0))
    tree = util.Tree(root = currentNode)
    
    fringe = util.Stack()
    fringe.push(currentNode)
    alreadyVisited = {fringe.peek().value[0]}
     # loop
    while(True):
        # if no candidates for expansion return failure
        if fringe.isEmpty():
            return "No path found to goal state"
        # choose leaf node for expansion according to the stack
        expanded = fringe.pop()
        # if node is the goal state return the solution
        if problem.isGoalState(expanded.value[0]):
            print(expanded.getPathToRoot())
            return expanded.getPathToRoot()
        else:
            # expand node and add the resulting nodes to the search tree
            print(f"expanding: {expanded.value}")
            for succ in problem.getSuccessors(expanded.value[0]):
                print(f"one successor: {succ}")
                nextState = succ[0]
                if nextState not in alreadyVisited:
                    nextNode = util.TreeNode(successorState = succ, parent = expanded)
                    expanded.addChild(nextNode)
                    fringe.push(nextNode)
                    alreadyVisited.add(nextState)
                
                # else if the node has been visited
                # else:
                #     # modify the parent to be the node that you're currently expanding
                #     childI = None
                #     try:
                #         childI = expanded.children.index(nextNode)
                #         expanded.children[childI].parent = nextNode
                #     except:
                #         print(f"exception: {nextNode.value[0]}")


# def depthFirstSearch(tree):
#     if tree.parent is None:
#         return []
#     else:
#         print()

def depthFirstSearch(problem):
    g = Graph(problem)
    g.dfsStart()
    return []

class Graph():
    def __init__(self, map):
        self.graph = {}
        self.start = map.startState
        self.goal = map.goal
        # build the graph from the map given
        if map is not None:
            print(map.getSuccessors(map.getStartState()))
            self.buildFromMap(map)

    def addNode(self, value):
        if value not in self.graph.keys():
            self.graph[value] = []
        else:
            print("this node already exists in the graph. can't add.")

    def buildFromMap(self, map):
        for x in range(map.walls.width):
            for y in range(map.walls.height):
                if not map.walls[x][y]:
                    self.addNode((x, y))
        
        for node in self.graph.keys():
            self.graph[node] = map.getSuccessors(node)
        
        print(self.graph)
        print(map.walls)

    def dfsStart(self):
        self.dfsSearchGoal(self.start, self.graph[self.start], visited = [])

    def dfsSearchGoal(self, node, connex, visited):
        visited.append(node)
        print(node, connex, visited)
        input("press enter to continue...")
        if node == self.goal:
            print("found goal")
        else:
            for next in connex:
                nextState = next[0]
                if nextState not in visited:
                    self.dfsSearchGoal(nextState, self.graph[nextState], visited)
                
            
        

        

    

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
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
