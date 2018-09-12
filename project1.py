import numpy as np
import pandas as pd
import scipy.stats as stats
from anytree import Node, NodeMixin, RenderTree
import sys
import os
import math

# USAGE: python3 project1.py training.csv testing.csv

################################
### GLOBAL VARIABLE SETTINGS ###
################################

# Choose between Gini Index or entropy for information gain
# Default is entropy, set GINI = 1 for Gini Index
GINI = 0

# Set confidence level for chi-square
# Choose between 0.99, 0.95, and 0 confidence level (99%, 95%, 0%)
CONFIDENCE_LEVEL = 0
ALPHA = 1 - CONFIDENCE_LEVEL

# Degrees of freedom for chi-square
# Dof = (classes-1)(values-1) = (3-1)(5-1) = 8
DOF = 8

# Critical value calculated from alpha and dof
CRITICAL_VALUE = stats.chi2.ppf(CONFIDENCE_LEVEL, DOF)

#######################
### FILE PROCESSING ###
#######################

# Given a path to a file, return it as a np array
def processFile(path):
    with open(path, 'r') as f:
        # Read into a numpy array
        a = np.genfromtxt(f, dtype=None, delimiter=',', encoding=None)
    return a

# Given a np array, return it as a file
def createFile(a):
    #np.savetxt("solution.csv", a, delimiter=',', fmt='<U21')
    df = pd.DataFrame(a)
    df.to_csv("solution.csv", header=None, index=None)

########################
### HELPER FUNCTIONS ###
########################

# Return an array with [countIE, countEI, countN]
def countPerClass(data):
    countIE = 0
    countEI = 0
    countN = 0
    for row in data:
        val = row[2]
        if val == "IE":
            countIE += 1
        elif val == "EI":
            countEI += 1
        elif val == "N":
            countN += 1
        else:
            countN += 1
    return [countIE, countEI, countN]

# Return the most common class in the data
# In case of a tie, TODO
def representativeClass(data):
    c = "N"
    result = countPerClass(data)
    countIE = result[0]
    countEI = result[1]
    countN = result[2]
    maximum = max(countIE, countEI, countN)
    # TODO: worry about tie cases?
    if countIE == maximum:
        c = "IE"
    elif countEI == maximum:
        c = "EI"
    else:
        c = "N"
    return c

# Return an array of arrays, split by value at pos
def decompose(data, pos):
    sampleRow = data[0]
    da = np.zeros_like(sampleRow)
    dc = np.zeros_like(sampleRow)
    dg = np.zeros_like(sampleRow)
    dt = np.zeros_like(sampleRow)
    dother = np.zeros_like(sampleRow)

    for row in data:
        val = row[1][pos] # get letter of DNA string at position pos
        if val == 'A':
            da = np.vstack((da, row))
        elif val == 'C':
            dc = np.vstack((dc, row))
        elif val == 'G':
            dg = np.vstack((dg, row))
        elif val == 'T':
            dt = np.vstack((dt, row))
        else:
            dother = np.vstack((dother, row))

    # Cut out first, initial row for each
    da = np.delete(da, 0)
    dc = np.delete(dc, 0)
    dg = np.delete(dg, 0)
    dt = np.delete(dt, 0)
    dother = np.delete(dother, 0)

    # Print for debugging purposes
    #print(da)

    d = np.array([ da, dc, dg, dt, dother ])

    return d

# Return true if any count equals the total number of examples, false otherwise
def isHomogeneous(counts, total):
    for count in counts:
        if count == total:
            return True
    return False

######################
### SPLIT CRITERIA ###
######################

def entropy(data):
    totalExamples = np.size(data, 0)
    # No division by 0
    if totalExamples == 0:
        return 0
    result = 0
    summary = countPerClass(data)
    for count in summary:
        proportion = count/totalExamples
        if proportion != 0: # No log of zero
            result -= proportion * math.log2(proportion)
    return result

def giniIndex(data):
    totalExamples = np.size(data, 0)
    # No division by 0
    if totalExamples == 0:
        return 0
    result = 0
    summary = countPerClass(data)
    for count in summary:
        proportion = count/totalExamples
        result += proportion * proportion
    return 1 - result

def infoGainE(data, att):
    gain = entropy(data)
    print("splitting data")
    splitData = decompose(data, att)
    totalExamples = np.size(data, 0) # number of rows in data
    for D in splitData:
        gain -= np.size(D, 0) / totalExamples * entropy(D)
    return gain

def infoGainG(data, att):
    gain = giniIndex(data)
    splitData = decompose(data, att)
    totalExamples = np.size(data, 0) # number of rows in data
    for D in splitData:
        gain -= np.size(D, 0) / totalExamples * giniIndex(D)
    return gain

def infoGain(data, att):
    if GINI:
        return infoGainG(data, att)
    return infoGainE(data, att)

def splitCriterion(data, attrs):
    print("Finding best split criterion")
    bestAtt = attrs.pop()
    attrs.add(bestAtt)
    bestIG = 0
    for att in attrs:
        ig = infoGain(data, att)
        if ig > bestIG:
            bestAtt = att
            bestIG = ig
    return [bestAtt, bestIG]

#####################
### STOP CRITERIA ###
#####################

def chiSquare(data, att):
    result = 0
    parentCounts = countPerClass(data)
    parentTotal = parentCounts[0] + parentCounts[1] + parentCounts[2]
    children = decompose(data, att)
    classes = [0, 1, 2]
    for c in classes:
        for child in children:
            childCounts = countPerClass(child)
            realCount = childCounts[c]
            childTotal = childCounts[0] + childCounts[1] + childCounts[2]
            expCount = childTotal * parentCounts[c]/ parentTotal
            # No division by 0 - don't count this if nothing in child
            if expCount != 0:
                result += (realCount - expCount)*(realCount - expCount)/expCount
    return result

def chiSquareTest(data, att):
    if CONFIDENCE_LEVEL == 0:
        return True
    print("running chi square on position ", att)
    chi = chiSquare(data, att)
    if chi > CRITICAL_VALUE:
        print(chi,": chi split")
        return True
    print(chi,": chi stop")
    return False

def impure(data):
    print("Checking impurity")
    numRows = np.size(data)
    # If empty, not impure
    if numRows == 0:
        return False
    counts = countPerClass(data)
    # If homogeneneous, not impure
    if isHomogeneous(counts, numRows):
        return False
    # Else impure
    return True

##########################
### ID3 TREE STRUCTURE ###
##########################

class baseNode(object):
    foo = 4

class id3Node(baseNode, NodeMixin):
    def __init__(self, parent=None, label="None", attr=[], value="",
    ig=0, isChild=False, isNull=False):
        self.parent = parent
        self.label = label
        self.attr = attr
        self.value = value
        self.ig = ig
        self.isChild = isChild
        self.isNull = isNull

def buildTree(data, parent, attrs, splitValue='', stop=False):
    t = id3Node(parent)
    t.label = representativeClass(data)
    t.value = splitValue
    if stop: # D is empty
        t.isNull = True
        return t
    criterion = 0
    if impure(data):
        if not attrs: # no more attributes to split off of
            t.isChild = True
            return t
        result = splitCriterion(data, attrs) #find position with most IG
        pos = result[0]
        # Do a chi square test to see if we should split on pos
        if not chiSquareTest(data, pos):
            t.isChild = True
            return t
        attrs.remove(pos) #remove pos from possible attrs
        t.ig = result[1]
        t.attr = pos
    else:
        t.isChild = True
        return t
    splitData = decompose(data, criterion)
    values = ['A','C','G','T','other']
    i = 0
    for D in splitData:
        if np.size(D,0) == 0: # D will be empty
            stop = True
        splitValue = values[i]
        buildTree(D, t, attrs, splitValue, stop) #build tree on split data with t as parent
        i += 1
    return t

######################
### CLASSIFICATION ###
######################

# Return label of this row given node of a decision tree
def classifyHelper(row, node):
    # Node is none, return most common class in tree, aka its label
    if node.isNull:
        label = node.label
    # Node is leaf
    elif node.isChild == True:
        label = node.label
    # Node is decision node
    else:
        # Find appropriate child node
        val = row[1][node.attr] # Find value of row at position attr
        for child in node.children:
            if child.value == val: # Check if split value in child matches
                newNode = child
                break
        # Run again on child node
        label = classifyHelper(row, newNode)
    return label

# Return a np array with format id,class
def classify(data, node):
    example = np.array([1,"EI"])
    result = np.zeros_like(example)
    for row in data:
        # Find label for this row
        label = classifyHelper(row, node)
        # Add label and id to result
        result = np.vstack((result, [row[0], label]))
    result[0] = ['id','class']
    return result

############
### MAIN ###
############

def main():
    # Get the training and testing file paths
    #trainingPath = os.path.abspath(sys.argv[1])
    #testingPath = sys.argv[2]

    # Hardcoded paths TODO: don't be hardcoded paths
    trainingPath = "/Users/rebeccasousa/Documents/UNM/CS529/Project1/training.csv"
    testingPath = "/Users/rebeccasousa/Documents/UNM/CS529/Project1/testing.csv"

    # Convert the training and testing files into numpy arrays
    trainingData = processFile(trainingPath)
    testingData = processFile(testingPath)

    # Print for testing purposes
    #print(trainingData)

    # Build the decision tree
    print("Building tree...")
    attrs = set(range(0, 60))
    dt = buildTree(trainingData, None, attrs)

    # Print for testing purposes
    #print(RenderTree(dt))

    # Classify the testing data
    print("Classifying testing data...")
    result = classify(testingData, dt)

    # Print for testing purposes
    #print(result)

    createFile(result)

    print("Classification successful.")

if __name__ == '__main__':
    main()
