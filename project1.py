import numpy as np
import pandas as pd
from anytree import Node, NodeMixin
import sys
import os

# USAGE: python3 project1.py training.csv testing.csv

### FILE PROCESSING ###
def processFile(path):
    f = open(path, "r")

    # Read into a numpy array
    a = np.genfromtxt(f, dtype=None, delimiter=',', encoding=None)

    f.close()
    return a

### HELPER FUNCTIONS ###

# Return the most common class in the data
def representativeClass(data):
    c = "N"
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
    maximum = max(countIE, countEI, countN)
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

### SPLIT CRITERIA ###

def giniIndex(data, att):
    result = 0
    # handle target attributes with arbitrary labels
    #dictionary = summarizeExamples(examples, targetAttribute)
    #for key in dictionary:
        #proportion = dictionary[key]/total number of examples
        #result += proportion * proportion
    return 1 - result

def entropy(data, att):
    result = 0
    # handle target attributes with arbitrary labels
    #dictionary = summarizeExamples(examples, targetAttribute)
    #for key in dictionary:
        #proportion = dictionary[key]/total number of examples
        #result -= proportion * log2(proportion)
    return result

def infoGain(data, att):
    gain = entropy(data)
    #for value in attributeValues(examples, attribute):
        #sub = subset(examples, attribute, value)
        #gain -=  (number in sub)/(total num of examples) * entropy(sub)
    return gain

def splitCriterion(data, attrs):
    bestAtt = attrs[0]
    bestIG = 0
    for att in attrs:
        ig = infoGain(data, att)
        if ig > bestIG:
            bestAtt = att
            bestIG = ig
    return [bestAtt, ig]

### STOP CRITERIA ###

def chiSquare():
    result = 0
    #for class in classes:
        #for value in attrValues:
            #result += (realCount - expCount)*(realCount - expCount)/expCount
    return result

### ID3 TREE STRUCTURE ###

class baseNode(object):
    foo = 4

class id3Node(baseNode, NodeMixin):
    def __init__(self, parent=None, my_class="None", attr=[], ig=0, chi=0):
        self.parent = parent
        self.my_class = my_class
        self.attr = attr
        self.ig = ig
        self.chi = chi

def buildTree(data, parent, attrs):
    t = id3Node(parent)
    t.my_class = representativeClass(data)
    #if(impure(data))
        #result = splitCriterion(data, attrs) #find position with most IG
        #criterion = result[0]
        #attrs.remove(criterion) #remove pos from possible attrs
        #t.ig = result[1]
        #t.attr = criterion
    #else
        #return t
    #splitData = decompose(data, criterion)
    #for D in splitData:
        #buildTree(D, t, attrs) #build tree on split data with t as parent
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
    attrs = range(0, 59)
    dt = buildTree(trainingData, None, attrs)

    # Print for testing purposes
    print(dt)

if __name__ == '__main__':
    main()
