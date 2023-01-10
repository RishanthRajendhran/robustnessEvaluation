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
    nargs="+",
    help = "Path to json file containing dataset",
    required=True
)

parser.add_argument(
    "-isDir",
    "--isDirectory",
    action="store_true",
    help="Booleaan flag to indicate if the -input input is a directory path",
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

parser.add_argument(
    "-pattern",
    help="RegEx pattern for json file names in the directory that needs to be used"
)

args = parser.parse_args()

inputFiles = args.input 
dataset = args.dataset
isDirectory = args.isDirectory
pattern = None
if args.pattern:
    pattern = args.pattern

if isDirectory:
    if not os.path.exists(inputFiles[0]):
        raise RuntimeError(f"{inputFiles[0]} is an invalid directory path!")
    jsonDirName = inputFiles[0]
    jsonPattern = os.path.join(jsonDirName, '*.json')
    inputFiles = glob.glob(jsonPattern)
    if pattern:
        try: 
            re.compile(pattern)
        except: 
            raise ValueError(f"{pattern} is not a valid regular expression!")
        inputFiles = [tf for tf in inputFiles if re.match(pattern, tf.split("/")[-1])]
        if len(inputFiles) == 0:
            raise RuntimeError(f"{pattern} did not match any file!")
        print(f"Chosen files: {inputFiles}")
else:
    for fl in inputFiles:
        file_exists = os.path.exists(fl)
        if not file_exists:
            raise RuntimeError(f"{fl} is an invalid input file path!")
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
def _generatePrompt(data, promptType, bestPromptType=1):
    prompts = []

    if promptType == 4:
        prompts.append("Answer the following yes/no/don’t know question by reasoning step by step.\n")
        promptType = bestPromptType
    elif promptType == 5:
        prompts.append("Give the rationale before answering.\n")
        promptType = bestPromptType

    for d in data:
        if promptType == 1 or promptType == 2 or promptType == 3:
            out = "Passage: "
            out += d["sentence1"]
            out += " Question: "
            out += d["sentence2"]
            out += "\nGive the rationale before answering. "
            if promptType == 2:
                out += "Answer: "
            if promptType == 3:
                out += "Lets think step by step."
            if "explanation" not in d.keys():
                raise Exception("Cannot do CoT prompting without explanations!")
            out += d["explanation"]
            out += " So the answer is "
            out += d["label"]
            out += ".\n"
            prompts.append(out)
    return prompts
#--------------------------------------------------------------------------------
def main():
    data = []
    for fl in inputFiles:
        data.extend(getData(fl))
    numInstances = len(data)
    avgContextLen = 0
    avgQuestionLen = 0
    avgAnswerLen = 0
    avgExpLen = 0 
    promptLen = 0
    avgPerQuePromptLen = 0
    if "sentence1" not in data[0].keys():
        raise RuntimeError("sentence1 is not a key!")
    if "sentence2" not in data[0].keys():
        raise RuntimeError("sentence2 is not a key!")
    if "label" not in data[0].keys():
        raise RuntimeError("label is not a key!")
    if "explanation" not in data[0].keys():
        raise RuntimeError("explanation is not a key!")
    for i in range(len(data)):
        avgContextLen += len(data[i]["sentence1"])
        avgQuestionLen += len(data[i]["sentence2"])
        avgAnswerLen += len(data[i]["label"])
        avgExpLen += len(data[i]["explanation"])
        avgPerQuePromptLen += len(_generatePrompt([data[i]], 1)[0])
    avgContextLen /= numInstances
    avgQuestionLen /= numInstances
    avgAnswerLen /= numInstances
    avgExpLen /= numInstances
    promptLen = avgPerQuePromptLen
    avgPerSampleSetPromptLen = avgPerQuePromptLen/len(inputFiles)
    avgPerQuePromptLen /= numInstances
    print(f"Samples Statistics:\nDataset: {dataset}")
    print(f"No. of files: {len(inputFiles)}")
    print(f"No. of data instances: {numInstances}")
    print(f"Average Context Length: {round(avgContextLen,2)}")
    print(f"Average Question Length: {round(avgQuestionLen,2)}")
    print(f"Average Answer Length: {round(avgAnswerLen,2)}")
    print(f"Average Explanation Length: {round(avgExpLen,2)}")
    print(f"Train Prompt (prompt type=1) Length: {promptLen}")
    print(f"Average Per Sample Set Prompt (prompt type=1) Length: {round(avgPerSampleSetPromptLen,2)}")
    print(f"Average Per Question Prompt (prompt type=1) Length: {round(avgPerQuePromptLen,2)}")
#--------------------------------------------------------------------------------
if __name__ == "__main__":
    main()






