import numpy as np
import pandas as pd
from anytree import Node, NodeMixin, RenderTree
import sys
import os
import math

# USAGE: python3 project1.py training.csv testing.csv

### FILE PROCESSING ###
def processFile(path):
    f = open(path, "r")

    # Read into a numpy array
    a = np.genfromtxt(f, dtype=None, delimiter=',', encoding=None)

    f.close()
    return a

### HELPER FUNCTIONS ###

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
    da = np.zeros_like(data)
    dc = np.zeros_like(data)
    dg = np.zeros_like(data)
    dt = np.zeros_like(data)
    dother = np.zeros_like(data)

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

    d = np.array([ da[1:], dc[1:], dg[1:], dt[1:], dother[1:] ])

    return d

def isHomogeneous(counts):
    a = counts[0]
    b = counts[1]
    c = counts[2]
    if a > 0:
        if b == 0 and c == 0:
            return True
        else:
            return False
    if b > 0:
        if a == 0 and c == 0:
            return True
        else:
            return False
    if c > 0:
        if a == 0 and b == 0:
            return True
        else:
            return False
    return False

### SPLIT CRITERIA ###

def entropy(data):
    result = 0
    summary = countPerClass(data)
    totalExamples = summary[0] + summary[1] + summary[2] #TODO: this or dimensions of data?
    for count in summary:
        proportion = count/totalExamples
        result -= proportion * math.log2(proportion)
    return result

def giniIndex(data):
    result = 0
    summary = countPerClass(data)
    totalExamples = summary[0] + summary[1] + summary[2] #TODO: this or dimensions of data?
    for count in summary:
        proportion = count/totalExamples
        result += proportion * proportion
    return 1 - result

def infoGain(data, att):
    gain = entropy(data)
    splitData = decompose(data, att)
    totalExamples = np.size(data, 0) # number of rows in data
    for D in splitData:
        gain -= np.size(D, 0) / totalExamples * entropy(D)
    return gain

def splitCriterion(data, attrs):
    bestAtt = 0
    bestIG = 0
    for att in attrs:
        ig = infoGain(data, att)
        if ig > bestIG:
            bestAtt = att
            bestIG = ig
    return [bestAtt, ig]

### STOP CRITERIA ###

def chiSquare(data):
    result = 0
    #for class in classes:
        #for value in attrValues:
            #result += (realCount - expCount)*(realCount - expCount)/expCount
    return result

def impure(data):
    # If empty, not impure
    if np.size(data) == 0:
        return False

    counts = countPerClass(data)

    # If homogeneneous, not impure
    if isHomogeneous(counts):
        return False

    # Else impure
    return True

    # TODO: implement chi-square for determining purity

### ID3 TREE STRUCTURE ###

class baseNode(object):
    foo = 4

class id3Node(baseNode, NodeMixin):
    def __init__(self, parent=None, label="None", attr=[], ig=0, chi=0):
        self.parent = parent
        self.label = label
        self.attr = attr
        self.ig = ig
        self.chi = chi

def buildTree(data, parent, attrs):
    t = id3Node(parent)
    t.label = representativeClass(data)
    criterion = 0
    if(impure(data)):
        result = splitCriterion(data, attrs) #find position with most IG
        criterion = result[0]
        attrs.remove(criterion) #remove pos from possible attrs
        t.ig = result[1]
        t.attr = criterion
    else:
        return t
    splitData = decompose(data, criterion)
    for D in splitData:
        buildTree(D, t, attrs) #build tree on split data with t as parent
    return t

### MAIN ###

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
    # use training data, dt is the root, attrs are from 0 to 59
    attrs = set(range(0, 59))
    dt = buildTree(trainingData, None, attrs)

    # Print for testing purposes
    print(dt.label)

    # Classify the testing data

if __name__ == '__main__':
    main()
