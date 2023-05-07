import argparse 
import json
import logging
import pandas as pd
from pathlib import Path
from os.path import exists

#------------------------------------------------------
def checkFile(fileName, fileExtension=None):
    if fileExtension:
        if not fileName.endswith(fileExtension):
            raise ValueError(f"{fileName} does not have expected file extension {fileExtension}!")
    file_exists = exists(fileName)
    if not file_exists:
        raise FileNotFoundError(f"{fileName} is an invalid file path!")
    path = Path(fileName)
    if not path.is_file():
        raise ValueError(f"{fileName} is not a file!")
#------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "-debug",
    action="store_true",
    help="Boolean flag to enable debug mode"
)

parser.add_argument(
    "-log",
    type=str,
    help="Path to file to print logging information",
    default=None
)

parser.add_argument(
    "-dataset",
    nargs = "+",
    choices = [
        "snli",
        "mnli"
    ],
    help="Name(s) of datasets for which predictions have been made"
)

parser.add_argument(
    "-predictions",
    help="Path to file containing predictions"
)

args = parser.parse_args()

debug = args.debug
logFile = args.log
datasets = args.dataset
predictionsFile = args.predictions

if logFile:
    checkFile(logFile)
    logging.basicConfig(filename=logFile, filemode='w', level=logging.INFO)
elif debug:
    logging.basicConfig(filemode='w', level=logging.DEBUG)
else:
    logging.basicConfig(filemode='w', level=logging.INFO)

checkFile(predictionsFile, ".jsonl")

with open(predictionsFile, 'r') as f:
    predictionsList = list(f)

predictions = []
for predictionStr in predictionsList:
    prediction = json.loads(predictionStr)
    predictions.append(prediction)
 
predsDF = pd.DataFrame(predictions)

allFileNames = predsDF["filename"].unique()

for fileName in allFileNames:
    curPredsDF = predsDF[predsDF["filename"] == fileName]
    
