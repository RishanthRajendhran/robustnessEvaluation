import json
import argparse
import numpy as np
import os
import glob
import logging
import regex as re
from conllu import parse

huggingfaceDatasets = [
    "boolqHF", 
    "condaqaHF",
    "quorefHF",
]

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
    help="Path to file containing original questions (Pass dummy json file for CondaQA/MATRES dataset)",
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
        "perspectrum",
        "udparsing",
        "boolqHF",
        "condaqaHF",
        "quorefHF",
    ],
    required=True,
)

parser.add_argument(
    "-debug",
    action="store_true",
    help="Boolean flag to turn on debug mode",
)

parser.add_argument(
    "-pattern",
    help="RegEx pattern for json file names in the output directory that need to be demarcated",
    default=None
)

parser.add_argument(
    "-selfConsistency",
    action="store_true",
    help="Boolean flag to enable self consistency mode"
)

parser.add_argument(
    "-numSamplePaths",
    type=int,
    help="No. of sampling paths in selfConsistency experiments",
    default=10
)

args = parser.parse_args()
outputFiles = args.outputFiles
isDirectory = args.isDirectory
dataset = args.dataset
originalFile = args.original
debug = args.debug
pattern=args.pattern
selfConsistency=args.selfConsistency
numSamplePaths=args.numSamplePaths

if isDirectory:
    jsonDirName = outputFiles[0]
    jsonPattern = os.path.join(jsonDirName, '*.json')
    outputFiles = glob.glob(jsonPattern)
    if pattern:
        try: 
            re.compile(pattern)
        except: 
            raise ValueError(f"{pattern} is not a valid regular expression!")
        outputFiles = [f for f in outputFiles if re.match(pattern, f)]
        if len(outputFiles) == 0:
            raise RuntimeError(f"{pattern} did not match any file!")
if dataset == "condaqa":
    pass
    # data = []
    # for line in open(originalFile,"r"):
    #     data.append(json.loads(line))
elif dataset == "boolq":
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
                if q["id"].strip() + " " + q["question"].strip() not in originalIDQs:
                    originalIDQs.append(q["id"].strip() + " " + q["question"].strip())
elif dataset == "ropes":
    with open(originalFile,"r", encoding='utf-8-sig') as f:
        original = json.load(f)
    original = original["data"][0]["paragraphs"]
elif dataset == "mctaco":
    with open(originalFile,"r", encoding='utf-8-sig') as f:
        original = json.load(f)
    originalIDs = {}
    for ori in original: 
        key = (ori["sentence1"].replace("\n","").strip() + " " + "Is this the answer: " + ori["sentence2"].replace("\n","").strip() + "?" + " " + ori["label"]).replace("\n","").strip()
        if key not in originalIDs.keys(): 
            originalIDs[key] = 0
elif dataset == "perspectrum":
    pass
    # with open(originalFile,"r", encoding='utf-8-sig') as f:
    #     data = json.load(f)
    # oriQues = {}
    # for d in data:
    #     if d["original_claim"] not in oriQues.keys():
    #         oriQues[d["oriQues"]] = True
elif dataset == "drop":
    with open(originalFile, "r") as f:
        data = json.load(f)
    oriQues = {}
    for k in data.keys():
        for q in data[k]["qa_pairs"]:
            if q["original_question"] not in oriQues.keys():
                oriQues[q["original_question"]] = True
elif dataset == "imdb":
    with open(originalFile, "r", encoding='utf-8-sig') as f:
        data = json.load(f)
    oriQues = {}
    for d in data:
        if "Text" not in d.keys() or "Sentiment" not in d.keys():
            continue
        if d["Text"] not in oriQues.keys():
            oriQues[d["Text"]] = d["Sentiment"]
        else: 
            logging.warning("{} already seen! This should not have happened.".format(d["Text"]))
elif dataset == "matres":
    pass
elif dataset == "udparsing":
    with open(originalFile, "r", encoding="utf-8") as f:
        dataOriginal = f.read()

    oriIDtexts = {}

    treesOri = parse(dataOriginal)
    for (i, treeOri) in enumerate(treesOri):
        sentID = treeOri.metadata["sent_id"]
        passage = ""
        for token in list(treeOri):
            if type(token["id"]) is list: 
                continue
            if not ((token["upos"] == "PUNCT") or (token["upos"] == "PART" and (token["xpos"] == "POS" or token["xpos"] == "RB"))):
                passage += " "
            passage += token["form"]
        passage = passage.replace("\n", "")
        passage = passage.strip()
        passage = re.sub("\*","",passage)

        newIDtext = sentID + "Sentence: " + passage
        if newIDtext not in oriIDtexts.keys():
            oriIDtexts[newIDtext] = 0
        if selfConsistency:
                oriIDtexts[newIDtext] += numSamplePaths
        else:
            oriIDtexts[newIDtext] += 1
else: 
    if dataset not in huggingfaceDatasets:
        logging.error(f"{dataset} is not a recognized dataset!")
        exit(0)

outputFiles.sort()
for outputFile in outputFiles:
    if dataset == "udparsing":
        oriIDtextsCopy = oriIDtexts.copy()
    if debug:
        logging.info(f"Marking File: {outputFile}")
    passages = []
    data = []   
    with open(outputFile, "r") as f:
        data = json.load(f)
    countOriginal = 0
    if dataset=="ropes":
        qIDmarked = []
    elif dataset=="mctaco":
        for k in originalIDs.keys():
            originalIDs[k] = False
    for d in data:
        d["label"] = str(d["label"])
        if ("output" in d.keys() and len(d["output"]) == 0) or ("label" in d.keys() and len(d["label"])) == 0:
            continue
        isOriginal = False
        if dataset not in huggingfaceDatasets:
            if dataset == "condaqa":
                if d["PassageEditID"] == 0:
                    isOriginal = True 
                    countOriginal += 1
            elif dataset == "boolq":
                for ori in original:
                    if ori["paragraph"] == d["sentence1"] and ori["question"] == d["sentence2"] and ori["answer"] == d["label"]:
                        isOriginal = True 
                        countOriginal += 1
                        break
            elif dataset == "quoref" and (d["QuestionID"].strip() + " " + d["sentence2"].strip()) in originalIDQs:
                isOriginal = True 
                countOriginal += 1 
            elif dataset == "ropes":
                for ori in original:
                    for q in ori["qas"]:
                        ansList = []
                        for k in range(len(q["answers"])):
                            for l in q["answers"][k].keys():
                                ansList.append(q["answers"][k][l])
                        ans = ", ".join(ansList)
                        if q["id"] == d["QuestionID"] and ori["background"].replace("\n", " ") == d["sentence1"] and (ori["situation"].replace("\n", " ") + " " + q["question"].replace("\n", " ")).strip()  == d["sentence2"] and ans.strip() == d["label"]:
                            if q["id"] not in qIDmarked:
                                isOriginal = True 
                                countOriginal += 1
                                if not selfConsistency:
                                    qIDmarked.append(q["id"])
                                break
            elif dataset == "mctaco":
                for ori in original: 
                    key = d["sentence1"] + " " + d["sentence2"] + " " + d["label"]
                    if key in originalIDs.keys() and (originalIDs[key] == 0 or selfConsistency):
                        originalIDs[key] += 1 
                        if originalIDs[key]<=numSamplePaths:
                            isOriginal = True 
                            countOriginal += 1
                        break
            elif dataset == "perspectrum":
                if int(d["QuestionID"])%2 == 1:
                # if d["sentence2"] in oriQues.keys():
                    isOriginal = True 
                    countOriginal += 1
            elif dataset == "drop":
                if d["sentence2"] in oriQues.keys():
                    isOriginal = True 
                    countOriginal += 1
            elif dataset == "imdb":
                if d["sentence1"] in oriQues.keys():
                    isOriginal = True 
                    countOriginal += 1
            elif dataset == "matres":
                if int(d["QuestionID"])%2 == 0:
                    isOriginal = True 
                    countOriginal += 1
            elif dataset == "udparsing":
                idText = "_".join(d["PassageID"].split("_")[1:-1]) + re.sub("\*","",d["sentence1"])
                if idText in oriIDtextsCopy.keys():
                    if oriIDtextsCopy[idText]:
                        oriIDtextsCopy[idText] -= 1
                        isOriginal = True 
                        countOriginal += 1
        newD = d.copy()
        newD.update({"isOriginal": isOriginal})
        passages.append(newD)
    newFilePath = "/".join(outputFile.split("/")[:-1]) + "/marked"
    newOutPath = newFilePath + "/" + outputFile.split("/")[-1]
    if not os.path.exists(newFilePath):
        os.makedirs(newFilePath)
    if debug:
        logging.info(f"\t Marked {countOriginal} examples as original out of {len(data)} examples")
        logging.info(f"\tWriting marked file into {newOutPath}")
    with open(newOutPath, "w") as f:
        json.dump(passages, f)