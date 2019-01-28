# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if action == 'Stop':
            return -float('inf')
        if successorGameState.isWin():
            return float('inf')

        oldFood = currentGameState.getFood()
        oldGhostStates = currentGameState.getGhostStates()

        capsuleplaces = currentGameState.getCapsules()
        score = -len(newFood.asList()) - len(capsuleplaces) * 5

        if (currentGameState.getNumFood() > successorGameState.getNumFood()):
            score += 100
        if successorGameState.getPacmanPosition() in capsuleplaces:
            score += 120

        capsuleplaceDist = [manhattanDistance(capsuleplace, newPos) for capsuleplace in capsuleplaces]
        distToClosestCapsuleplace = min(capsuleplaceDist) if len(capsuleplaceDist) > 0 else 0
        score += 10 / max(distToClosestCapsuleplace, 1)

        foodDist = [manhattanDistance(food, newPos) for food in newFood.asList()]
        distToClosestFood = min(foodDist)
        score += 10 / max(distToClosestFood, 1)
        foodDist = [manhattanDistance(food, newPos) for food in oldFood.asList()]
        distToClosestFood = min(foodDist)
        score += 8 / max(distToClosestFood, 1)
        averageDistToFood = sum(foodDist) / len(foodDist)
        score += 3 / max(averageDistToFood, 1)
        distToFurestFood = max(foodDist)
        score += 5 / max(distToFurestFood, 1)

        distanceToGhost = [manhattanDistance(ghostState.getPosition(), newPos) for ghostState in oldGhostStates if ghostState.scaredTimer == 0]
        minDistanceToGhost = min(distanceToGhost) if len(distanceToGhost) > 0 else 0
        score += - (25 ** (3 - min(3, minDistanceToGhost)))

        return score

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.pacmanIndex = 0

    def isTerminal(self, gameState, depth):
        return gameState.isWin() or gameState.isLose() or self.depth == depth

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        bestAction, bestScore = self.maxValue(gameState, 0)
        return bestAction

    def maxValue(self, gameState, currentDepth):
        if self.isTerminal(gameState, currentDepth):
            return ('None', self.evaluationFunction(gameState))

        bestScore = float('-inf')
        actions = gameState.getLegalActions(0)
        for action in actions:
            state = gameState.generateSuccessor(0, action)
            _, minValue = self.minValue(state, currentDepth, 1)

            if minValue > bestScore:
                bestAction = action
                bestScore = minValue

        return bestAction, bestScore

    def minValue(self, gameState, currentDepth, currentAgentIndex):
        if currentAgentIndex >= gameState.getNumAgents():
            return self.maxValue(gameState, currentDepth + 1)

        if self.isTerminal(gameState, currentDepth):
            return ('None', self.evaluationFunction(gameState))

        bestScore = float('inf')
        actions = gameState.getLegalActions(currentAgentIndex)
        for action in actions:
            state = gameState.generateSuccessor(currentAgentIndex, action)
            _, minValue = self.minValue(state, currentDepth, currentAgentIndex + 1)

            if minValue < bestScore:
                bestAction = action
                bestScore = minValue

        return bestAction, bestScore


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        bestAction, bestScore = self.maxValue(gameState, 0, float('-inf'), float('inf'))
        return bestAction

    def maxValue(self, gameState, currentDepth, alpha, beta):
        if self.isTerminal(gameState, currentDepth):
            return ('None', self.evaluationFunction(gameState))

        bestScore = float('-inf')
        actions = gameState.getLegalActions(0)
        for action in actions:
            state = gameState.generateSuccessor(0, action)
            _, minValue = self.minValue(state, currentDepth, 1, alpha, beta)

            if minValue > bestScore:
                bestAction = action
                bestScore = minValue

            if minValue > beta:
                return bestAction, bestScore
            alpha = max(alpha, bestScore)

        return bestAction, bestScore

    def minValue(self, gameState, currentDepth, currentAgentIndex, alpha, beta):
        if currentAgentIndex >= gameState.getNumAgents():
            return self.maxValue(gameState, currentDepth + 1, alpha, beta)

        if self.isTerminal(gameState, currentDepth):
            return ('None', self.evaluationFunction(gameState))

        bestScore = float('inf')
        actions = gameState.getLegalActions(currentAgentIndex)
        for action in actions:
            state = gameState.generateSuccessor(currentAgentIndex, action)
            _, minValue = self.minValue(state, currentDepth, currentAgentIndex + 1, alpha, beta)

            if minValue < bestScore:
                bestAction = action
                bestScore = minValue

            if minValue < alpha:
                return bestAction, bestScore
            beta = min(beta, bestScore)

        return bestAction, bestScore

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()

        bestAction, bestScore = self.maxValue(gameState, 0)
        return bestAction

    def maxValue(self, gameState, currentDepth):
        if self.isTerminal(gameState, currentDepth):
            return ('None', self.evaluationFunction(gameState))

        bestScore = float('-inf')
        actions = gameState.getLegalActions(0)
        for action in actions:
            state = gameState.generateSuccessor(0, action)
            expValue = self.expValue(state, currentDepth, 1)

            if expValue > bestScore:
                bestAction = action
                bestScore = expValue

        return bestAction, bestScore


    def expValue(self, gameState, currentDepth, currentAgentIndex):
        if currentAgentIndex >= gameState.getNumAgents():
            action, score = self.maxValue(gameState, currentDepth + 1)
            return score

        if self.isTerminal(gameState, currentDepth):
            return self.evaluationFunction(gameState)

        bestScore = 0
        actions = gameState.getLegalActions(currentAgentIndex)
        probability = 1 / len(actions)
        for action in actions:
            state = gameState.generateSuccessor(currentAgentIndex, action)
            minValue = self.expValue(state, currentDepth, currentAgentIndex + 1)

            bestScore += minValue * probability

        return bestScore

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
