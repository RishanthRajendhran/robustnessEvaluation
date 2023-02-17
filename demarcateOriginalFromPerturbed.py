import json
import argparse
import numpy as np
import os
import glob
import logging

logging.basicConfig(filemode="w", level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument(
    "-out",
    "--outputFiles",
    nargs = "+",
    required = True,
    help="List of output file names/Path to directory containing output json files (Need to set -isDir flag)"
)

parser.add_argument(
    "-isDir",
    "--isDirectory",
    action="store_true",
    help="Booleaan flag to indicate if the -out input is a directory path"
)

parser.add_argument(
    "-original",
    help="Path to file containing original questions",
    required=True
)

parser.add_argument(
    "-dataset",
    choices=[
        "condaqa",
        "boolq",
        "ropes",
        "drop",
        "quoref",
        "mctaco",
        "imdb",
        "matres",
        "perspectrum"
    ],
    required=True,
)

parser.add_argument(
    "-debug",
    action="store_true",
    help="Boolean flag to turn on debug mode",
)

args = parser.parse_args()
outputFiles = args.outputFiles
isDirectory = args.isDirectory
dataset = args.dataset
originalFile = args.original
debug = args.debug

if isDirectory:
    jsonDirName = outputFiles[0]
    jsonPattern = os.path.join(jsonDirName, '*.json')
    outputFiles = glob.glob(jsonPattern)

if dataset == "boolq":
    with open(originalFile, "r") as f:
        original = json.load(f)
    original = original["data"][1:]
elif dataset == "quoref":
    with open(originalFile,"r", encoding='utf-8-sig') as f:
        original = json.load(f)
    original = original["data"]
    originalIDQs = []
    for d in original: 
        for c in d["paragraphs"]:
            for q in c["qas"]:
                if q["id"] + " " + q["question"] not in originalIDQs:
                    originalIDQs.append(q["id"] + " " + q["question"])

outputFiles.sort()
for outputFile in outputFiles:
    if debug:
        logging.info(f"Marking File: {outputFile}")
    passages = []
    data = []   
    with open(outputFile, "r") as f:
        data = json.load(f)
    for d in data:
        if len(d["output"]) == 0 or len(d["label"]) == 0:
            continue
        isOriginal = False
        for ori in original:
            if dataset == "boolq" and ori["paragraph"] == d["sentence1"] and ori["question"] == d["sentence2"] and ori["answer"] == d["label"]:
                isOriginal = True 
                break
            elif dataset == "quoref" and (d["QuestionID"] + " " + d["sentence2"]) in originalIDQs:
                isOriginal = True 
                break 
        newD = d.copy()
        newD.update({"isOriginal": isOriginal})
        passages.append(newD)
    newFilePath = "/".join(outputFile.split("/")[:-1]) + "/new"
    newOutPath = newFilePath + "/" + outputFile.split("/")[-1]
    if not os.path.exists(newFilePath):
        os.makedirs(newFilePath)
    if debug:
        logging.info(f"\tWriting marked file into {newOutPath}")
    with open(newOutPath, "w") as f:
        json.dump(passages, f)