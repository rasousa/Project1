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

### SPLIT CRITERIA ###

def giniIndex():
    result = 0
    # handle target attributes with arbitrary labels
    #dictionary = summarizeExamples(examples, targetAttribute)
    #for key in dictionary:
        #proportion = dictionary[key]/total number of examples
        #result += proportion * proportion
    return 1 - result

def entropy(data):
    result = 0
    # handle target attributes with arbitrary labels
    #dictionary = summarizeExamples(examples, targetAttribute)
    #for key in dictionary:
        #proportion = dictionary[key]/total number of examples
        #result -= proportion * log2(proportion)
    return result

def infoGain(data):
    gain = entropy(data)
    #for value in attributeValues(examples, attribute):
        #sub = subset(examples, attribute, value)
        #gain -=  (number in sub)/(total number of examples) * entropy(sub)
    return gain

### STOP CRITERIA ###

#def chiSquare():
    # TODO

### ID3 TREE STRUCTURE ###

class baseNode(object):
    foo = 4

class id3Node(baseNode, NodeMixin):
    def __init__(self, parent=None, classify="None", attr=[], ig=0, chi=0):
        self.parent = parent
        self.classify = classify
        self.attr = attr
        self.ig = ig
        self.chi = chi

def buildTree(data, parent):
    t = id3Node(parent)
    #label(t) = representativeClass(data)
    #if(impure(data))
        #criterion = splitCriterion(data)
    #else
        #return t
    #[D1, D2, ..., Dn] = decomposing(data, criterion)
    #for D in [D]:
        #buildTree(D, t) #build tree on split data with t as parent
    #return t

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
    dt = buildTree(trainingData, None)

    # Print for testing purposes
    print(dt)

if __name__ == '__main__':
    main()
