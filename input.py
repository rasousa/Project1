import os
import sys

def main():
    try:
        trainingPath = os.path.abspath(sys.argv[1])
    except:
        print("Error: not enough arguments")
        sys.exit(1)
    print(trainingPath)

if __name__ == '__main__':
    main()
