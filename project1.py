#import pandas as pd
import numpy as np
import sys
import os

# USAGE: python project1.py training.csv testing.csv

def processFile(path):
    f = open(path, "r")
    #line1 = f.readline()

    # Read into a numpy array
    a = np.genfromtxt(f, dtype=None, delimiter=',')
    f.close()
    return a

#def buildTree():
    # TODO

#def calcGini():
    # TODO

#def calcIG():
    # TODO

#def chiSquare():
    # TODO

def main():
    # Get the training and testing file paths
    #trainingPath = os.path.abspath(sys.argv[1])
    #testingPath = sys.argv[2]

    # Hardcoded paths
    trainingPath = "/Users/rebeccasousa/Documents/UNM/CS529/Project1/training.csv"
    testingPath = "/Users/rebeccasousa/Documents/UNM/CS529/Project1/testing.csv"

    # Convert the training and testing files into numpy arrays
    trainingData = processFile(trainingPath)
    testingData = processFile(testingPath)

    print(trainingData)
    print(testingData)

if __name__ == '__main__':
    main()
