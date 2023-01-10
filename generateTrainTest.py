import argparse 
import json
import numpy as np
import os
from pathlib import Path
import glob
import re

parser = argparse.ArgumentParser()

parser.add_argument(
    "-input",
    help = "Path to json file containing dataset",
    required=True
)

parser.add_argument(
    "-outTrain",
    help = "directory path to save output train file",
    required=True
)

parser.add_argument(
    "-outTest",
    help = "directory path to save output test file",
    required=True
)

parser.add_argument(
    "-numSamples",
    type=int,
    help="Number of question-answer samples to pick for train set",
    required=True,
)

parser.add_argument(
    "-dataset",
    choices = ["condaqa", 
                "boolq", 
                "drop", 
                "ropes", 
                "mctaco", 
                "quoref", 
                "imdb", 
                "matres", 
                "nlvr2",
                "perspectrum"
            ],
    required=True
)

args = parser.parse_args()

inputFile = args.input 
outputTrainPath = args.outTrain
outputTestPath = args.outTest
numSamples = args.numSamples
dataset = args.dataset

file_exists = os.path.exists(inputFile)
if not file_exists:
    raise RuntimeError(f"{inputFile} is an invalid input file path!")

if not os.path.exists(outputTrainPath):
    raise RuntimeError(f"{outputTrainPath} is not a valid directory path!")

if not os.path.exists(outputTestPath):
    raise RuntimeError(f"{outputTestPath} is not a valid directory path!")
    
#--------------------------------------------------------------------------------
def getData(inputFile):
    if dataset == "condaqa":
        data = []
        for line in open(inputFile,"r"):
            data.append(json.loads(line))
    else:
        with open(inputFile, "r") as f:
            data = json.load(f)
    return data
#--------------------------------------------------------------------------------
def pickSamples(data, ns):
    passages = {}
    for d in data:
        if d["PassageID"] not in passages.keys():
            passages[d["PassageID"]] = []
        passages[d["PassageID"]].append(d)
    passKeys = np.array(list(passages.keys()))
    chosenIndices = np.random.choice(len(passKeys), size=ns, replace=False)
    train = []
    test = []
    for i in range(len(passKeys)):
        if i in chosenIndices:
            chosenQA = np.random.randint(len(passages[passKeys[i]]))
            train.append(passages[passKeys[i]][chosenQA])
            test.extend(passages[passKeys[i]][:chosenQA])
            test.extend(passages[passKeys[i]][chosenQA+1:])
        else:
            test.extend(passages[passKeys[i]])
    return train, test
#--------------------------------------------------------------------------------
def main():
    data = getData(inputFile)
    train, test = pickSamples(data, numSamples)

    jsonObj = json.dumps(train, indent=4)
    with open(outputTrainPath+"_"+inputFile.split("/")[-1].split(".")[0]+"_"+"train"+".json","w") as o:
        o.write(jsonObj)
    
    jsonObj = json.dumps(test, indent=4)
    with open(outputTestPath+"_"+inputFile.split("/")[-1].split(".")[0]+"_"+"test"+".json","w") as o:
        o.write(jsonObj)
#--------------------------------------------------------------------------------
if __name__ == "__main__":
    main()






