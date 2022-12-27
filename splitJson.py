import argparse
import json
import sys
from os.path import exists

parser = argparse.ArgumentParser()

parser.add_argument(
    "-input",
    help="Path to input json file",
    required=True,
)

parser.add_argument(
    "-output",
    help="Path to output json file",
    required=True,
)

parser.add_argument(
    "-split",
    type=int,
    help="Number of splits of input file that is needed (w/o replacement)",
    default=1,
)

args = parser.parse_args()
inputFile = args.input
outputFile = None
if args.output:
    outputFile = args.output
numSplit = args.split
#------------------------------------
def main():
    if not inputFile.endswith(".json") or not exists(inputFile):
        raise Exception("-input argmument invalid!")
    if not outputFile.endswith(".json"):
        raise Exception("-output argmument invalid!")
    examples = []
    for line in open(inputFile,"r"):
        examples.append(json.loads(line))
    global numSplit
    while(1):
        splitSize = len(examples)//numSplit
        if splitSize*numSplit != len(examples):
            print("Warning: Input split does not give an even split")
            print(f"Number of examples: {len(examples)}")
            print("Do you want to change the number of splits? (y/n) ", end="")
            decision = input()
            if decision == "n":
                break 
            print("Enter the number of splits: ", end="")
            numSplit = int(input())
        else: 
            break
    for s in range(numSplit):
        newOutputFile = outputFile.split(".json")[0] + "_" + str(s) + ".json"
        newExamples = examples[s*splitSize:(s+1)*splitSize].copy()
        #If this is the last split, add all remaining examples to this split
        if s == numSplit-1:
            newExamples = examples[s*splitSize:]
        if numSplit == 1:
            newOutputFile = outputFile
        with open(newOutputFile,"w") as of:
            json.dump(newExamples, of, indent=4)
#------------------------------------
if __name__ == "__main__":
    main()