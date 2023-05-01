import json
from os.path import exists
from pathlib import Path
import argparse
import torch
import transformers
import os
import glob
import re
import logging

logging.basicConfig(filemode="w",level=logging.INFO)

NUM_SEQUENCES = 10
TEMPERATURE = 0.7
MAX_LENGTH = 500

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
    choices = [1, 2, 3, 4, 5, 6],
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
    choices = [
        "condaqa", 
        "boolq", 
        "ropes",
        "drop",
        "quoref",
        "mctaco",
        "imdb",
        "matres",
        "perspectrum",
        "udparsing"
    ],
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

parser.add_argument(
    "-selfConsistency",
    action="store_true",
    help="Boolean flag to enable self consistency mode"
)

parser.add_argument(
    "-noCoT",
    action="store_true",
    help="Boolean flag to indicate no-Chain-of-Thought inferencing",
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
noCoT = args.noCoT
trainPattern = None 
testPattern = None
selfConsistency = args.selfConsistency
if args.trainPattern:
    trainPattern = args.trainPattern
if args.testPattern:
    testPattern = args.testPattern

if noCoT:
    if promptType != 1:
        raise ValueError("Only promptType 1 supported in no-Chain-of-thought inferencing mode!")
    if selfConsistency:
        raise ValueError("Self consisency not supported in no-Chain-of-thought inferencing mode!")

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
        with open(filePath, "r") as f:
            data = json.load(f)
        return data
# #---------------------------------------------------------------------------
def _generatePrompt(data, promptType, dataset, noCoT, bestPromptType=1, isTest=False):
    prompts = []

    ##Experiment: For Boolq
    # prompts.append("In this task, you’re expected to write answers to questions using “yes”, “no” or “unknown”. Give the rationale before answering.\n")
    if promptType == 4:
        if not isTest:
            prompts.append("Answer the following yes/no/don’t know question by reasoning step by step.\n")
        promptType = bestPromptType
    elif promptType == 5:
        if not isTest:
            prompts.append("Give the rationale before answering.\n")
        promptType = bestPromptType
    elif promptType == 6:
        prompts.append("In this task, you’re expected to write answers to questions involving reasoning about negation. The answer to the question should be “yes”, “no”, “don’t know” or a phrase in the passage. Questions can have only one correct answer. Give the rationale before answering.\n")
    for d in data:
        if promptType == 1 or promptType == 2 or promptType == 3 or promptType == 6:
            out = ""
            if dataset not in ["perspectrum", "udparsing"]:
                if dataset == "imdb":
                    out += "Review: "
                else:
                    out += "Passage: "
            out += d["sentence1"]
            out += "\n"
            if dataset != "perspectrum":
                if  dataset != "mctaco" and dataset != "imdb":
                    out += "Question: "
                out += d["sentence2"]
            else: 
                reqInd1 = d["sentence2"].index("Does the perspective support or undermine the claim?")
                reqInd2 = d["sentence2"].index("Answer with: supports, undermines or not a valid perspective.")
                out += d["sentence2"][:reqInd1] + "\n" + d["sentence2"][reqInd1:reqInd2] + "\n" + d["sentence2"][reqInd2:]
            if not noCoT:
                if promptType != 6:
                    out += "\nGive the rationale before answering. "
                if promptType == 2:
                    out += "Answer: "
                if promptType == 3:
                    out += "Answer: Lets think step by step. "
                if promptType == 6:
                    out += "\nAnswer: Lets think step by step. "
                if not isTest:
                    if "explanation" not in d.keys():
                        raise Exception("Cannot do CoT prompting without explanations!")
                    out += d["explanation"]
                    out += " So the answer is "
            elif not isTest:
                out += "\nThe answer is "
            if not isTest:
                out += d["label"]
                out += ".\n###\n"
            prompts.append(out)
    return prompts
#---------------------------------------------------------------------------
def generateTrainPrompt(data, promptType, dataset, noCoT, bestPromptType=1):
    return _generatePrompt(data,promptType, dataset, noCoT, bestPromptType)
#---------------------------------------------------------------------------
def generateTestPrompt(data, promptType, dataset, noCoT, bestPromptType=1):
    return _generatePrompt([data],promptType, dataset, noCoT, bestPromptType, True)[0]
#---------------------------------------------------------------------------

#Check if file exists
for trainFile in trainFiles:
    file_exists = exists(trainFile)
    if not file_exists:
        raise ValueError(f"{trainFile} is an invalid train file path!")
    path = Path(trainFile)
    if not path.is_file():
        raise ValueError(f"{trainFile} is not a (train) file!")

#Check if file exists
for testFile in testFiles:
    file_exists = exists(testFile)
    if not file_exists:
        raise ValueError(f"{testFile} is an invalid test file path!")
    path = Path(testFile)
    if not path.is_file():
        raise ValueError(f"{testFile} is not a (test) file!")

#Checking if trainOuts and a directory for this dataset exist, if it doesnt, create them
if not os.path.exists(f"./testOuts"):
    os.makedirs(f"./testOuts")
if selfConsistency:
    if not os.path.exists(f"./testOuts/selfConsistency"):
        os.makedirs(f"./testOuts/selfConsistency")
    if not os.path.exists(f"./testOuts/selfConsistency/{dataset}"):
        os.makedirs(f"./testOuts/selfConsistency/{dataset}")
else: 
    if not os.path.exists(f"./testOuts/{dataset}"):
        os.makedirs(f"./testOuts/{dataset}")

#Only do inferencing
with torch.no_grad():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    if selfConsistency:
        pipe_flan = transformers.pipeline(
            "text2text-generation", 
            model=f"google/flan-t5-{modelSize}", 
            device=device, 
            model_kwargs= {
                "torch_dtype":torch.bfloat16, 
                "num_return_sequences":NUM_SEQUENCES, 
                "do_sample":True,
                "temperature": TEMPERATURE
            }
        )
    else:
        pipe_flan = transformers.pipeline(
            "text2text-generation", 
            model=f"google/flan-t5-{modelSize}", 
            device=device, 
            model_kwargs= {
                "torch_dtype":torch.bfloat16
            }
        )
    # pipe_flan = None
    logging.info(f"Model: FLANT5-{modelSize}")
    if selfConsistency:
        logging.info("Decoding: Self Consistency")
    else:
        logging.info("Decoding: Greedy")
    if noCoT:
        logging.info("Chain of Thought Prompting: False")
    else: 
        logging.info("Chain of Thought Prompting: True")
    for trainFile in trainFiles:
        #Contingency
        #Remove after first successful run
        logging.info(f"#{trainFile}")
        #---------------------------------
        if not zeroShot:
            trainData = readJSON(trainFile, dataset)
            trainPrompt = generateTrainPrompt(trainData, promptType, dataset, noCoT, bestPromptType)
        for testFile in testFiles:
            #Contingency
            #Remove after first successful run
            logging.info(f"##{testFile}")
            #---------------------------------
            testData = readJSON(testFile, dataset)
            testOuts = []
            testInd = -1
            for testEx in testData:
                testInd +=1
                testPrompt = generateTestPrompt(testEx, promptType, dataset, noCoT, bestPromptType)
                if not zeroShot:
                    finalPrompt = ("".join(trainPrompt)) + testPrompt
                else:
                    finalPrompt = testPrompt                        

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if selfConsistency:
                    output_flan = pipe_flan(finalPrompt, max_length=MAX_LENGTH)

                    sampleID = f"{trainFile}_{testFile}_{testInd}"

                    for out in output_flan: 
                        newEx = testEx.copy()
                        newEx["SampleID"] = sampleID
                        newEx["output"] = out["generated_text"]
                        testOuts.append(newEx)
                else:
                    output_flan = pipe_flan(finalPrompt, max_length=MAX_LENGTH)[0]["generated_text"]
                    
                    testEx["output"] = output_flan
                    testOuts.append(testEx)                           

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if noCoT:
                with open(f'./testOuts/noCoT/{dataset}/{trainFile.split("/")[-1].split(".")[0]}__{testFile.split("/")[-1].split(".")[0]}__{promptType}__{zeroShot}.json', 'w') as fout:
                    json.dump(testOuts , fout)
            else:
                if selfConsistency:
                    with open(f'./testOuts/selfConsistency/{dataset}/{trainFile.split("/")[-1].split(".")[0]}__{testFile.split("/")[-1].split(".")[0]}__{promptType}__{zeroShot}.json', 'w') as fout:
                        json.dump(testOuts , fout)
                else:
                    with open(f'./testOuts/{dataset}/{trainFile.split("/")[-1].split(".")[0]}__{testFile.split("/")[-1].split(".")[0]}__{promptType}__{zeroShot}.json', 'w') as fout:
                        json.dump(testOuts , fout)
        logging.info("*****")
    logging.info("-----")
