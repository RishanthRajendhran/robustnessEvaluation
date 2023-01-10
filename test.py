import json
from os.path import exists
from pathlib import Path
import argparse
import torch
import transformers
import os
import glob
import re

parser = argparse.ArgumentParser()

parser.add_argument(
    "-train",
    "--trainFiles",
    nargs = "+",
    required = True
)

parser.add_argument(
    "-test",
    "--testFiles",
    nargs = "+",
    required = True
)

parser.add_argument(
    "-isTrainDir",
    "--isTrainDirectory",
    action="store_true",
    help="Booleaan flag to indicate if the -train input is a directory path",
)

parser.add_argument(
    "-isTestDir",
    "--isTestDirectory",
    action="store_true",
    help="Booleaan flag to indicate if the -test input is a directory path",
)

parser.add_argument(
    "-promptType",
    type = int,
    default = 1,
    choices = [1, 2, 3, 4, 5],
)

parser.add_argument(
    "-bestPromptType",
    type = int,
    default = 1,
    choices = [1, 2, 3],
    help = "When promptType is set as 4/5, bestPromptType defines the sub-prompt type",
)

parser.add_argument(
    "-zeroShot",
    action="store_true",
)

parser.add_argument(
    "-dataset",
    choices = ["condaqa", "boolq", "ropes"],
    required = True,
)

parser.add_argument(
    "-modelSize",
    default="xxl"
)

parser.add_argument(
    "-trainPattern",
    help="RegEx pattern for json file names in the train directory that need to be used"
)

parser.add_argument(
    "-testPattern",
    help="RegEx pattern for json file names in the test directory that need to be merged"
)

args = parser.parse_args()

trainFiles = args.trainFiles
testFiles = args.testFiles
promptType = args.promptType
bestPromptType = args.bestPromptType
zeroShot = args.zeroShot
isTrainDirectory = args.isTrainDirectory
isTestDirectory = args.isTestDirectory
dataset = args.dataset
modelSize = args.modelSize
trainPattern = None 
testPattern = None
if args.trainPattern:
    trainPattern = args.trainPattern
if args.testPattern:
    testPattern = args.testPattern

if isTrainDirectory:
    jsonDirName = trainFiles[0]
    jsonPattern = os.path.join(jsonDirName, '*.json')
    trainFiles = glob.glob(jsonPattern)
    if trainPattern:
        try: 
            re.compile(trainPattern)
        except: 
            raise ValueError(f"{trainPattern} is not a valid regular expression!")
        trainFiles = [tf for tf in trainFiles if re.match(trainPattern, tf)]
        if len(trainFiles) == 0:
            raise RuntimeError(f"{trainPattern} did not match any file!")

if isTestDirectory:
    jsonDirName = testFiles[0]
    jsonPattern = os.path.join(jsonDirName, '*.json')
    testFiles = glob.glob(jsonPattern)
    if testPattern:
        try: 
            re.compile(testPattern)
        except: 
            raise ValueError(f"{testPattern} is not a valid regular expression!")
        testFiles = [tf for tf in testFiles if re.match(testPattern, tf)]
        if len(testFiles) == 0:
            raise RuntimeError(f"{testPattern} did not match any file!")

#---------------------------------------------------------------------------
def readJSON(filePath, dataset):
    if dataset == "condaqa":
        data = []
        for line in open(filePath,"r"):
            data.append(json.loads(line))
        return data
    else: 
        f = open(filePath, "r")
        data = json.load(f)
        return data
# #---------------------------------------------------------------------------
def _generatePrompt(data, promptType, bestPromptType=1, isTest=False):
    prompts = []

    if promptType == 4:
        if not isTest:
            prompts.append("Answer the following yes/no/don’t know question by reasoning step by step.\n")
        promptType = bestPromptType
    elif promptType == 5:
        if not isTest:
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
            if not isTest:
                if "explanation" not in d.keys():
                    raise Exception("Cannot do CoT prompting without explanations!")
                out += d["explanation"]
                out += " So the answer is "
                out += d["label"]
                out += ".\n"
            prompts.append(out)
    return prompts
#---------------------------------------------------------------------------
# def _generatePrompt(data, promptType, bestPromptType=1, isTest=False):
#     prompts = []

#     if promptType == 4:
#         if not isTest:
#             prompts.append("Answer the following yes/no/don’t know question by reasoning step by step.\n")
#         promptType = bestPromptType
#     elif promptType == 5:
#         if not isTest:
#             prompts.append("Give the rationale before answering.\n")
#         promptType = bestPromptType
#     passages = {}
#     for d in data:
#         if d["sentence1"] not in passages.keys():
#             passages[d["sentence1"]] = []
#         exp = ""
#         if "explanation" in d.keys():
#             exp = d["explanation"]
#         curQue = {
#             "sentence2": d["sentence2"],
#             "label": d["label"],
#             "explanation": exp
#         }
#         passages[d["sentence1"]].append(curQue)
#     for p in passages.keys():
#         if promptType == 1 or promptType == 2 or promptType == 3:
#             out = "Passage: "
#             out += p
#             out += " Answer the following questions based on this passage:\n"
#             for q in passages[p]:
#                 out += "Question: "
#                 out += q["sentence2"]
#                 out += "\nGive the rationale before answering. "
#                 if promptType == 2:
#                     out += "Answer: "
#                 if promptType == 3:
#                     out += "Lets think step by step."
#                 if not isTest:
#                     if "explanation" not in d.keys():
#                         raise Exception("Cannot do CoT prompting without explanations!")
#                     out += d["explanation"]
#                     out += " So the answer is "
#                     out += d["label"]
#                     out += ".\n"
#             prompts.append(out)
#     return prompts
#---------------------------------------------------------------------------
def generateTrainPrompt(data, promptType, bestPromptType=1):
    return _generatePrompt(data,promptType, bestPromptType)
#---------------------------------------------------------------------------
def generateTestPrompt(data, promptType, bestPromptType=1):
    return _generatePrompt([data],promptType, bestPromptType, True)[0]
#---------------------------------------------------------------------------

#Check if file exists
for trainFile in trainFiles:
    file_exists = exists(trainFile)
    if not file_exists:
        print(f"{trainFile} is an invalid train file path!")
        exit(0)
    path = Path(trainFile)
    if not path.is_file():
        print(f"{trainFile} is not a (train) file!")
        exit(0)

#Check if file exists
for testFile in testFiles:
    file_exists = exists(testFile)
    if not file_exists:
        print(f"{testFile} is an invalid test file path!")
        exit(0)
    path = Path(testFile)
    if not path.is_file():
        print(f"{testFile} is not a (test) file!")
        exit(0)

#Checking if trainOuts and a directory for this dataset exist, if it doesnt, create them
if not os.path.exists(f"./testOuts"):
    os.makedirs(f"./testOuts")
if not os.path.exists(f"./testOuts/{dataset}"):
    os.makedirs(f"./testOuts/{dataset}")

#Only do inferencing
with torch.no_grad():
    pipe_flan = transformers.pipeline("text2text-generation", model=f"google/flan-t5-{modelSize}", device="cuda:0", model_kwargs={"torch_dtype":torch.bfloat16})
    print(f"Model: FLANT5-{modelSize}")
    for trainFile in trainFiles:
        #Contingency
        #Remove after first successful run
        print(f"#{trainFile}")
        #---------------------------------
        if not zeroShot:
            trainData = readJSON(trainFile, dataset)
            trainPrompt = generateTrainPrompt(trainData, promptType, bestPromptType)
        for testFile in testFiles:
            #Contingency
            #Remove after first successful run
            print(f"##{testFile}")
            #---------------------------------
            testData = readJSON(testFile, dataset)
            testOuts = []
            for testEx in testData:
                testPrompt = generateTestPrompt(testEx, promptType, bestPromptType)
                if not zeroShot:
                    finalPrompt = ("".join(trainPrompt)) + testPrompt
                else:
                    finalPrompt = testPrompt                           

                torch.cuda.empty_cache()
                
                output_flan = pipe_flan(finalPrompt, max_length=100)[0]["generated_text"]
                
                testEx["output"] = output_flan
                testOuts.append(testEx)                           

                torch.cuda.empty_cache()
                
                # del testPrompt
                # del finalPrompt
                # del testEx
                # del output_flan
                # gc.collect()
                # torch.cuda.empty_cache()

                # #Contingency
                # #Remove after first successful run
                # print(testEx)
                #---------------------------------

            with open(f'./testOuts/{dataset}/{trainFile.split("/")[-1].split(".")[0]}__{testFile.split("/")[-1].split(".")[0]}__{promptType}__{zeroShot}.json', 'w') as fout:
                json.dump(testOuts , fout)
        print("*****")
    print("-----")