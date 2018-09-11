import numpy as np
import pandas as pd
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

def chiSquare(data):
    result = 0
    #for class in classes:
        #for value in attrValues:
            #result += (realCount - expCount)*(realCount - expCount)/expCount
    return result

def impure(data):
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

    # TODO: implement chi-square for determining purity

##########################
### ID3 TREE STRUCTURE ###
##########################

class baseNode(object):
    foo = 4

class id3Node(baseNode, NodeMixin):
    def __init__(self, parent=None, label="None", attr=[], value="", ig=0, chi=0, isChild=False):
        self.parent = parent
        self.label = label
        self.attr = attr
        self.value = value
        self.ig = ig
        self.chi = chi
        self.isChild = isChild

def buildTree(data, parent, attrs):
    t = id3Node(parent)
    t.label = representativeClass(data)
    criterion = 0
    if impure(data):
        if not attrs: # no more attributes to split off of
            t.isChild = True
            return t
        result = splitCriterion(data, attrs) #find position with most IG
        criterion = result[0]
        attrs.remove(criterion) #remove pos from possible attrs
        t.ig = result[1]
        t.attr = criterion
    else:
        t.isChild = True
        return t
    splitData = decompose(data, criterion)
    for D in splitData:
        buildTree(D, t, attrs) #build tree on split data with t as parent
    return t

######################
### CLASSIFICATION ###
######################

# Return label of this row given node of a decision tree
def classifyHelper(row, node):
    # Node is none, return most common class in tree
    if not node:
        label = representativeClass(data)
    # Node is leaf
    elif node.isChild == True:
        label = node.label
    # Node is decision node
    else:
        # Find appropriate child node
        child = 0
        val = row[1][node.attr]
        if val == 'A':
            child = 0
        elif val == 'C':
            child = 1
        elif val == 'G':
            child = 2
        elif val == 'T':
            child = 3
        else:
            child = 4
        newNode = node.children[child]
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
    attrs = set(range(0, 60))
    dt = buildTree(trainingData, None, attrs)

    # Print for testing purposes
    #print(RenderTree(dt))

    # Classify the testing data
    result = classify(testingData, dt)

    print(result)

    createFile(result)

if __name__ == '__main__':
    main()
