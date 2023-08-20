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
import numpy as np
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaModel, LlamaConfig
from datasets import load_dataset
import deepspeed
from deepspeed.ops.transformer.inference import DeepSpeedTransformerInference
from tqdm import tqdm

logging.basicConfig(filemode="w",level=logging.INFO)

huggingfaceDatasetsNames = {
    "boolqHF": "boolq",
    "condaqaHF": "lasha-nlp/CONDAQA",
    "quorefHF": "quoref",
    "ropesHF": "ropes",
}
huggingfaceDatasets = list(huggingfaceDatasetsNames.keys())

NUM_SEQUENCES = 10
TEMPERATURE = 0.7
MAX_LENGTH = 512

NUM_SEQUENCES_MPT = 10
TEMPERATURE_MPT = 0.1
MAX_LENGTH_MPT = 512
MAX_NEW_TOKENS_MPT = 256

NUM_SEQUENCES_ALPACA = 10
TEMPERATURE_ALPACA = 0.7
MAX_LENGTH_ALPACA = 4096
MAX_NEW_TOKENS_ALPACA = 256
TOP_P_ALPACA=0.1
TOP_K_ALPACA=40
DO_SAMPLE_ALPACA=True 
USE_CACHE_ALPACA=True
REPETITION_PENALTY_ALPACA=1.176

NUM_SEQUENCES_LLAMA = 10
TEMPERATURE_LLAMA = 0.7
TOP_P_LLAMA=0.1
TOP_K_LLAMA=40
REPETITION_PENALTY_LLAMA=1.176
# TEMPERATURE_LLAMA = 0.72
# TOP_P_LLAMA=0.73
# TOP_K_LLAMA=0
# REPETITION_PENALTY_LLAMA=1.1
MAX_LENGTH_LLAMA = 4096
MAX_NEW_TOKENS_LLAMA = 128
DO_SAMPLE_LLAMA=True
USE_CACHE_LLAMA=True
MAX_CONTEXT_LENGTH_LLAMA=2048

parser = argparse.ArgumentParser()

parser.add_argument(
    "-train",
    "--trainFiles",
    nargs = "+",
    default=["train"]
)

parser.add_argument(
    "-test",
    "--testFiles",
    nargs = "+",
    default=["validation"]
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
        "udparsing",
        "boolqHF",
        "condaqaHF",
        "quorefHF",
    ],
    required = True,
)

parser.add_argument(
    "-model",
    choices=["flant5", "mpt", "alpaca", "llama"],
    help="Name of model to use for inference",
    default="flant5"
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

parser.add_argument(
    "-maxShots",
    type=int,
    help="Maximum number of shots to use in few-shot setting",
    default=9
)

#Arguments for DeepSpeed
parser.add_argument(
    "--local_rank", 
    type=int, 
    help="[DEEPSPEED ARGUMENT]",
    default=0
)

parser.add_argument(
    "--do_eval",
    action="store_true",
    help="[DEEPSPEED ARGUMENT] Boolean flag to enable inference mode"
)

parser.add_argument(
    "--deepspeed", 
    help="[DEEPSPEED ARGUMENT] Path to deepspeed configuration"
)

args = parser.parse_args()

dataset = args.dataset
noCoT = args.noCoT
if args.trainFiles:
    trainFiles = args.trainFiles
# elif dataset in huggingfaceDatasets:
#     trainFiles = ["train"]
else:
    raise  ValueError("Train path not provided!")
if args.testFiles:
    testFiles = args.testFiles
# elif dataset in huggingfaceDatasets:
#     testFiles = ["validation"]
else:
    raise  ValueError("Test path not provided!")
promptType = args.promptType
bestPromptType = args.bestPromptType
zeroShot = args.zeroShot
isTrainDirectory = args.isTrainDirectory
isTestDirectory = args.isTestDirectory
model = args.model
modelSize = args.modelSize
trainPattern = None 
testPattern = None
selfConsistency = args.selfConsistency
maxShots = args.maxShots
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

#Check if file exists
for trainFile in trainFiles:
    if not trainFile.endswith(".json") and dataset in huggingfaceDatasets:
        continue
    file_exists = exists(trainFile)
    if not file_exists:
        raise ValueError(f"{trainFile} is an invalid train file path!")
    path = Path(trainFile)
    if not path.is_file():
        raise ValueError(f"{trainFile} is not a (train) file!")
#Check if file exists
for testFile in testFiles:
    if not testFile.endswith(".json") and dataset in huggingfaceDatasets:
        continue
    file_exists = exists(testFile)
    if not file_exists:
        raise ValueError(f"{testFile} is an invalid test file path!")
    path = Path(testFile)
    if not path.is_file():
        raise ValueError(f"{testFile} is not a (test) file!")

if dataset in huggingfaceDatasets and not noCoT and sum([1 if fileName.endswith(".json") else 0 for fileName in trainFiles])!=len(trainFiles):
    raise ValueError("CoT not supported for HuggingFace datasets!")

#---------------------------------------------------------------------------
def readJSON(filePath, dataset):
    if dataset == "condaqa" or dataset == "condaqaHF":
        data = []
        for line in open(filePath,"r"):
            data.append(json.loads(line))
        return data
    else: 
        with open(filePath, "r") as f:
            data = json.load(f)
        return data
# #---------------------------------------------------------------------------
def _generatePrompt(data, promptType, dataset, noCoT, model, maxShots, bestPromptType=1, isTest=False):
    prompts = []

    overallPrompts = {
        "default": {
            4: {
                "default":"Answer the following yes/no/don’t know question by reasoning step by step.\n",
                "isTest": ""
            },
            5: {
                "default": "Give the rationale before answering.\n",
                "isTest": ""
            },
            6: {
                "default": "In this task, you’re expected to write answers to questions involving reasoning about negation. The answer to the question should be “yes”, “no”, “don’t know” or a phrase in the passage. Questions can have only one correct answer. Give the rationale before answering.\n",
                "isTest": ""
            },
            #Default Case
            -1: {
                "default": "",
                "isTest": ""
            },
        },
        "alpaca": {
            #Default Case
            -1: {
                "default": "",
                "isTest": "",
                # "boolq": {
                #     "default": "Answer the questions with yes or no\n",
                #     "isTest": ""
                # }
            },
        },
        "llama": {
            #Default Case
            -1: {
                "default": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n",
                "isTest": "",
                # "boolq": {
                #     "default": "Answer the questions with yes/no/don't know\n",
                #     "isTest": ""
                # }
            },
        }
    }
    individualPrompts = {
        "default": {
            1: {
                "CoT": {
                    "default": "Passage: {passage}\nQuestion: {question}\nGive the rationale before answering. {explanation} So the answer is {answer}.\n###\n",
                    "imdb": "Review: {review}\n{question}\nGive the rationale before answering. {explanation} So the answer is {answer}.\n###\n",
                    "perspectrum": "{perspective}\n{claim}\n{question}\n{instruction}\nGive the rationale before answering. {explanation} So the answer is {answer}.\n###\n",
                    "mctaco": "Passage: {passage}\n{question}\nGive the rationale before answering. {explanation} So the answer is {answer}.\n###\n",
                },
                "noCoT": {
                    "default": "Passage: {passage}\nQuestion: {question}\nThe answer is {answer}.\n###\n",
                    "imdb": "Review: {passage}\n{question}\nThe answer is {answer}.\n###\n",
                    "perspectrum": "{perspective}\n{claim}\n{question}\n{instruction}\nThe answer is {answer}.\n###\n",
                    "mctaco": "Passage: {passage}\n{question}\nThe answer is {answer}.\n###\n",
                    "udparsing": "{sentence}\nQuestion: {question}\nThe answer is {answer}.\n###\n",
                },
                "isTest": {
                    "CoT": {
                        "default": "Passage: {passage}\nQuestion: {question}\nGive the rationale before answering.",
                        "imdb": "Review: {review}\n{question}\nGive the rationale before answering.",
                        "perspectrum": "{perspective}\n{claim}\n{question}\n{instruction}\nGive the rationale before answering.",
                        "mctaco": "Passage: {passage}\n{question}\nGive the rationale before answering.",
                    },
                    "noCoT": {
                        "default": "Passage: {passage}\nQuestion: {question}",
                        "imdb": "Review: {review}\n{question}",
                        "perspectrum": "{perspective}\n{claim}\n{question}\n{instruction}",
                        "mctaco": "Passage: {passage}\n{question}",
                        "udparsing": "{sentence}\nQuestion: {question}",
                    },
                }
            }, 
            2: {
                "CoT": {
                    "default": "Passage: {passage}\nQuestion: {question}\nGive the rationale before answering. Answer: {explanation} So the answer is {answer}.\n###\n",
                    "imdb": "Review: {review}\n{question}\nGive the rationale before answering. Answer: {explanation} So the answer is {answer}.\n###\n",
                    "perspectrum": "{perspective}\n{claim}\n{question}\n{instruction}\nGive the rationale before answering. Answer: {explanation} So the answer is {answer}.\n###\n",
                    "mctaco": "Passage: {passage}\n{question}\nGive the rationale before answering. Answer: {explanation} So the answer is {answer}.\n###\n",
                },
                "noCoT": {
                    "default": "Passage: {passage}\nQuestion: {question}\nAnswer: The answer is {answer}.\n###\n",
                    "imdb": "Review: {passage}\n{question}\nAnswer: The answer is {answer}.\n###\n",
                    "perspectrum": "{perspective}\n{claim}\n{question}\n{instruction}\nAnswer: The answer is {answer}.\n###\n",
                    "mctaco": "Passage: {passage}\n{question}\nAnswer: The answer is {answer}.\n###\n",
                    "udparsing": "{sentence}\nQuestion: {question}\nAnswer: The answer is {answer}.\n###\n",
                },
                "isTest": {
                    "CoT": {
                        "default": "Passage: {passage}\nQuestion: {question}\nGive the rationale before answering. Answer: ",
                        "imdb": "Review: {review}\n{question}\nGive the rationale before answering. Answer: ",
                        "perspectrum": "{perspective}\n{claim}\n{question}\n{instruction}\nGive the rationale before answering. Answer: ",
                        "mctaco": "Passage: {passage}\n{question}\nGive the rationale before answering. Answer: ",
                    },
                    "noCoT": {
                        "default": "Passage: {passage}\nQuestion: {question} Answer: ",
                        "imdb": "Review: {review}\n{question} Answer: ",
                        "perspectrum": "{perspective}\n{claim}\n{question}\n{instruction} Answer: ",
                        "mctaco": "Passage: {passage}\n{question} Answer: ",
                        "udparsing": "{sentence}\nQuestion: {question} Answer: ",
                    },
                }
            },
            3: {
                "CoT": {
                    "default": "Passage: {passage}\nQuestion: {question}\nGive the rationale before answering. Answer: Let's think step by step. {explanation} So the answer is {answer}.\n###\n",
                    "imdb": "Review: {review}\n{question}\nGive the rationale before answering. Answer: Let's think step by step. {explanation} So the answer is {answer}.\n###\n",
                    "perspectrum": "{perspective}\n{claim}\n{question}\n{instruction}\nGive the rationale before answering. Answer: Let's think step by step. {explanation} So the answer is {answer}.\n###\n",
                    "mctaco": "Passage: {passage}\n{question}\nGive the rationale before answering. Answer: Let's think step by step. {explanation} So the answer is {answer}.\n###\n",
                },
                "noCoT": {
                    "default": "Passage: {passage}\nQuestion: {question}\nAnswer: Let's think step by step. The answer is {answer}.\n###\n",
                    "imdb": "Review: {passage}\n{question}\nAnswer: Let's think step by step. The answer is {answer}.\n###\n",
                    "perspectrum": "{perspective}\n{claim}\n{question}\n{instruction}\nAnswer: Let's think step by step. The answer is {answer}.\n###\n",
                    "mctaco": "Passage: {passage}\n{question}\nAnswer: Let's think step by step. The answer is {answer}.\n###\n",
                    "udparsing": "{sentence}\nQuestion: {question}\nAnswer: Let's think step by step. The answer is {answer}.\n###\n",
                },
                "isTest": {
                    "CoT": {
                        "default": "Passage: {passage}\nQuestion: {question}\nGive the rationale before answering. Answer: Let's think step by step. ",
                        "imdb": "Review: {review}\n{question}\nGive the rationale before answering. Answer: Let's think step by step. ",
                        "perspectrum": "{perspective}\n{claim}\n{question}\n{instruction}\nGive the rationale before answering. Answer: Let's think step by step. ",
                        "mctaco": "Passage: {passage}\n{question}\nGive the rationale before answering. Answer: Let's think step by step. ",
                    },
                    "noCoT": {
                        "default": "Passage: {passage}\nQuestion: {question} Answer: Let's think step by step. ",
                        "imdb": "Review: {review}\n{question} Answer: Let's think step by step. ",
                        "perspectrum": "{perspective}\n{claim}\n{question}\n{instruction} Answer: Let's think step by step. ",
                        "mctaco": "Passage: {passage}\n{question} Answer: Let's think step by step. ",
                        "udparsing": "{sentence}\nQuestion: {question} Answer: Let's think step by step. ",
                    },
                }
            },
        }, 
        "mpt": {
            1: {
                "CoT": {
                    "default": "{passage}\nQuestion: {question}?\nGive the rationale before answering. Answer: {explanation} So the answer is {answer}.\n",
                },
                "noCoT": {
                    "default": "{passage}\nQuestion: {question}?\nAnswer: The answer is {answer}.\n",
                },
                "isTest": {
                    "CoT": {
                        "default": "{passage}\nQuestion: {question}?\nGive the rationale before answering. Answer: ",
                    },
                    "noCoT": {
                        "default": "{passage}\nQuestion: {question}?\nAnswer: ",
                    },
                }
            }, 
        },
        "alpaca": {
            1: {
                "CoT": {
                    "default": "Passage: {passage}\nQuestion: {question}?\nGive the rationale before answering.\nAnswer: {explanation} So the answer is {answer}.\n\n",
                },
                "noCoT": {
                    # "default": "{passage}\nQuestion: {question}?\nAnswer: The answer is {answer}.\n",
                    # "default": "{passage}\nQuestion: {question}?\nAnswer: {answer}\n",
                    "default": "Passage: {passage}\nQuestion: {question}?\nAnswer: The answer is {answer}\n\n",
                },
                "isTest": {
                    "CoT": {
                        "default": "Passage: {passage}\nQuestion: {question}?\nGive the rationale before answering.\nAnswer: ",
                    },
                    "noCoT": {
                        "default": "Passage: {passage}\nQuestion: {question}?\nAnswer: ",
                        # "boolq": "Passage: {passage}\nQuestion: {question} Yes or No?\nAnswer: ",
                        # "default": "Passage: {passage}\nQuestion: {question}?\n",
                    },
                }
            }, 
        },
        "llama": {
            1: {
                "CoT": {
                    "default": "### Passage:\n{passage}\n### Question:\n{question}?\n### Response:\n### Explanation:\n{explanation}\n### Correct Answer:\n{answer}\n\n",
                    "imdb": "### Review:\n{review}\n### Question:\n{question}\n### Response:\n### Explanation:\n{explanation}\n### Correct Answer:\n{answer}\n\n",
                    "perspectrum": "### Perspective:\n{perspective}\n### Claim:\n{claim}\n### Question:\n{question}\n### Instruction:\n{instruction}\n### Response:\n### Explanation:\n{explanation}\n### Correct Answer:\n{answer}\n\n",
                },
                "noCoT": {
                    "default": "### Passage:\n{passage}\n### Question:\n{question}?\n### Response:\n{answer}\n\n",
                    "imdb": "### Review:\n{review}\n### Question:\n{question}\n### Response:\n{answer}\n\n",
                    "perspectrum": "### Perspective:\n{perspective}\n### Claim:\n{claim}\n### Question:\n{question}\n### Instruction:\n{instruction}\n### Response:\n{answer}\n\n",
                    "udparsing": "### Sentence:\n{sentence}\n### Question:\n{question}\n### Response:\n{answer}\n\n",
                },
                "isTest": {
                    "CoT": {
                        "default": "### Passage:\n{passage}\n### Question:\n{question}?\n### Response:\n",
                        "imdb": "### Review:\n{review}\n### Question:\n{question}\n### Response:\n",
                        "perspectrum": "### Perspective:\n{perspective}\n### Claim:\n{claim}\n### Question:\n{question}\n### Instruction:\n{instruction}\n### Response:\n",
                    },
                    "noCoT": {
                        "default": "### Passage:\n{passage}\n### Question:\n{question}?\n### Response:\n",
                        "imdb": "### Review:\n{review}\n### Question:\n{question}\n### Response:\n",
                        "perspectrum": "### Perspective:\n{perspective}\n### Claim:\n{claim}\n### Question:\n{question}\n### Instruction:\n{instruction}\n### Response:\n",
                        "udparsing": "### Sentence:\n{sentence}\n### Question:\n{question}\n### Response:\n",
                    },
                }
            }, 
        },
    }

    if model not in overallPrompts.keys():
        if promptType in overallPrompts["default"].keys():
            if dataset in overallPrompts["default"][promptType].keys():
                if not isTest:
                    prompts.append(overallPrompts["default"][promptType][dataset]["default"])
                else: 
                    prompts.append(overallPrompts["default"][promptType][dataset]["isTest"])
            else:
                if not isTest:
                    prompts.append(overallPrompts["default"][promptType]["default"])
                else: 
                    prompts.append(overallPrompts["default"][promptType]["isTest"])
        else: 
            if dataset in overallPrompts["default"][-1].keys():
                if not isTest:
                    prompts.append(overallPrompts["default"][-1][dataset]["default"])
                else: 
                    prompts.append(overallPrompts["default"][-1][dataset]["isTest"])
            else:
                if not isTest:
                    prompts.append(overallPrompts["default"][-1]["default"])
                else: 
                    prompts.append(overallPrompts["default"][-1]["isTest"])
    else: 
        if promptType in overallPrompts[model].keys():
            if dataset in overallPrompts[model][promptType].keys():
                if not isTest:
                    prompts.append(overallPrompts[model][promptType][dataset]["default"])
                else: 
                    prompts.append(overallPrompts[model][promptType][dataset]["isTest"])
            else:
                if not isTest:
                    prompts.append(overallPrompts[model][promptType]["default"])
                else: 
                    prompts.append(overallPrompts[model][promptType]["isTest"])
        else: 
            if dataset in overallPrompts[model][-1].keys():
                if not isTest:
                    prompts.append(overallPrompts[model][-1][dataset]["default"])
                else: 
                    prompts.append(overallPrompts[model][-1][dataset]["isTest"])
            else:
                if not isTest:
                    prompts.append(overallPrompts[model][-1]["default"])
                else: 
                    prompts.append(overallPrompts[model][-1]["isTest"])

    if promptType not in [1,2,3,6]:
        promptType = bestPromptType
    
    if model not in individualPrompts.keys():
        model = "default"
    if promptType not in individualPrompts[model].keys():
        raise ValueError(f"{promptType} not a recognized type for {model}!")

    for i, d in enumerate(data):
        if i >= maxShots:
            break
        if isTest:
            innerKey = "isTest"
            if noCoT:
                innerInnerKey = "noCoT"
            else: 
                if dataset == "udparsing":
                    raise ValueError("CoT not supported for udparsing!")
                innerInnerKey = "CoT"
            if dataset not in individualPrompts[model][promptType][innerKey][innerInnerKey].keys():
                curPrompt = individualPrompts[model][promptType][innerKey][innerInnerKey]["default"].format(
                    passage=d["sentence1"],
                    question=d["sentence2"],
                )
            # elif dataset == "ropesHF":
            #     curPrompt = individualPrompts[model][promptType][innerKey][innerInnerKey][dataset].format(
            #         background=d["background"],
            #         situation=d["situation"],
            #         question=d["question"],
            #     )
            elif dataset == "imdb":
                curPrompt = individualPrompts[model][promptType][innerKey][innerInnerKey][dataset].format(
                    review=d["sentence1"],
                    question=d["sentence2"],
                )
            elif dataset == "perspectrum":
                reqInd1 = d["sentence2"].index("Does the perspective support or undermine the claim?")
                reqInd2 = d["sentence2"].index("Answer with: supports, undermines or not a valid perspective.")
                curPrompt = individualPrompts[model][promptType][innerKey][innerInnerKey][dataset].format(
                    perspective=d["sentence1"],
                    claim=d["sentence2"][:reqInd1],
                    question=d["sentence2"][reqInd1:reqInd2],
                    instruction=d["sentence2"][reqInd2:],
                )
            elif dataset == "mctaco":
                curPrompt = individualPrompts[model][promptType][innerKey][innerInnerKey][dataset].format(
                    passage=d["sentence1"],
                    question=d["sentence2"],
                )
            elif dataset == "udparsing":
                curPrompt = individualPrompts[model][promptType][innerKey][innerInnerKey][dataset].format(
                    sentence=d["sentence1"],
                    question=d["sentence2"],
                )
        else: 
            if noCoT:
                innerKey = "noCoT"
                if dataset not in individualPrompts[model][promptType][innerKey].keys():
                    curPrompt = individualPrompts[model][promptType][innerKey]["default"].format(
                        passage=d["sentence1"],
                        question=d["sentence2"],
                        answer=d["label"],
                    )
                # elif dataset == "ropesHF":
                #     curPrompt = individualPrompts[model][promptType][innerKey][dataset].format(
                #         background=d["background"],
                #         situation=d["situation"],
                #         question=d["question"],
                #         answer=d["answer"],
                #     )
                elif dataset == "imdb":
                    curPrompt = individualPrompts[model][promptType][innerKey][dataset].format(
                        review=d["sentence1"],
                        question=d["sentence2"],
                        answer=d["label"],
                    )
                elif dataset == "perspectrum":
                    reqInd1 = d["sentence2"].index("Does the perspective support or undermine the claim?")
                    reqInd2 = d["sentence2"].index("Answer with: supports, undermines or not a valid perspective.")
                    curPrompt = individualPrompts[model][promptType][innerKey][dataset].format(
                        perspective=d["sentence1"],
                        claim=d["sentence2"][:reqInd1],
                        question=d["sentence2"][reqInd1:reqInd2],
                        instruction=d["sentence2"][reqInd2:],
                        answer=d["label"],
                    )
                elif dataset == "mctaco":
                    curPrompt = individualPrompts[model][promptType][innerKey][dataset].format(
                        passage=d["sentence1"],
                        question=d["sentence2"],
                        answer=d["label"],
                    )
                elif dataset == "udparsing":
                    curPrompt = individualPrompts[model][promptType][innerKey][dataset].format(
                        sentence=d["sentence1"],
                        question=d["sentence2"],
                        answer=d["label"],
                    )
            else: 
                if dataset == "udparsing":
                    raise ValueError("CoT not supported for udparsing!")
                innerKey = "CoT"
                if dataset not in individualPrompts[model][promptType][innerKey].keys():
                    curPrompt = individualPrompts[model][promptType][innerKey]["default"].format(
                        passage=d["sentence1"],
                        question=d["sentence2"],
                        explanation=d["explanation"],
                        answer=d["label"],
                    )
                # elif dataset == "ropesHF":
                #     curPrompt = individualPrompts[model][promptType][innerKey][dataset].format(
                #         background=d["background"],
                #         situation=d["situation"],
                #         question=d["question"],
                #         answer=d["answer"],
                #         explanation=d["explanation"]
                #     )
                elif dataset == "imdb":
                    curPrompt = individualPrompts[model][promptType][innerKey][dataset].format(
                        review=d["sentence1"],
                        question=d["sentence2"],
                        explanation=d["explanation"],
                        answer=d["label"],
                    )
                elif dataset == "perspectrum":
                    reqInd1 = d["sentence2"].index("Does the perspective support or undermine the claim?")
                    reqInd2 = d["sentence2"].index("Answer with: supports, undermines or not a valid perspective.")
                    curPrompt = individualPrompts[model][promptType][innerKey][dataset].format(
                        perspective=d["sentence1"],
                        claim=d["sentence2"][:reqInd1],
                        question=d["sentence2"][reqInd1:reqInd2],
                        instruction=d["sentence2"][reqInd2:],
                        explanation=d["explanation"],
                        answer=d["label"],
                    )
                elif dataset == "mctaco":
                    curPrompt = individualPrompts[model][promptType][innerKey][dataset].format(
                        passage=d["sentence1"],
                        question=d["sentence2"],
                        explanation=d["explanation"],
                        answer=d["label"],
                    )
                elif dataset == "udparsing":
                    curPrompt = individualPrompts[model][promptType][innerKey][dataset].format(
                        sentence=d["sentence1"],
                        question=d["sentence2"],
                        explanation=d["explanation"],
                        answer=d["label"],
                    )
        prompts.append(curPrompt)
    return prompts
#---------------------------------------------------------------------------
def generateTrainPrompt(data, promptType, dataset, noCoT, model, maxShots=np.inf, bestPromptType=1):
    return _generatePrompt(data,promptType, dataset, noCoT, model, maxShots, bestPromptType)
#---------------------------------------------------------------------------
def generateTestPrompt(data, promptType, dataset, noCoT, model, maxShots=np.inf, bestPromptType=1):
    # return _generatePrompt([data],promptType, dataset, noCoT, model, maxShots, bestPromptType, True)[0]
    return _generatePrompt([data],promptType, dataset, noCoT, model, maxShots, bestPromptType, True)
#---------------------------------------------------------------------------
#The function below is from https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
#---------------------------------------------------------------------------
#Checking if trainOuts and a directory for this dataset exist, if it doesnt, create them
if not os.path.exists(f"./testOuts"):
    os.makedirs(f"./testOuts")
if not os.path.exists(f"./testOuts/{model}"):
    os.makedirs(f"./testOuts/{model}")
if zeroShot:
    if not os.path.exists(f"./testOuts/{model}/zeroShot"):
        os.makedirs(f"./testOuts/{model}/zeroShot")
    if selfConsistency:
        if not os.path.exists(f"./testOuts/{model}/zeroShot/selfConsistency"):
            os.makedirs(f"./testOuts/{model}/zeroShot/selfConsistency")
        if noCoT:
            if not os.path.exists(f"./testOuts/{model}/zeroShot/selfConsistency/noCoT"):
                os.makedirs(f"./testOuts/{model}/zeroShot/selfConsistency/noCoT")
            if not os.path.exists(f"./testOuts/{model}/zeroShot/selfConsistency/noCoT/{dataset}"):
                os.makedirs(f"./testOuts/{model}/zeroShot/selfConsistency/noCoT/{dataset}")
        else:
            if not os.path.exists(f"./testOuts/{model}/zeroShot/selfConsistency/{dataset}"):
                os.makedirs(f"./testOuts/{model}/zeroShot/selfConsistency/{dataset}")
    else: 
        if noCoT:
            if not os.path.exists(f"./testOuts/{model}/zeroShot/noCoT"):
                os.makedirs(f"./testOuts/{model}/zeroShot/noCoT")
            if not os.path.exists(f"./testOuts/{model}/zeroShot/noCoT/{dataset}"):
                os.makedirs(f"./testOuts/{model}/zeroShot/noCoT/{dataset}")
        else:
            if not os.path.exists(f"./testOuts/{model}/zeroShot/{dataset}"):
                os.makedirs(f"./testOuts/{model}/zeroShot/{dataset}")
else:
    if not os.path.exists(f"./testOuts/{model}/fewShot"):
        os.makedirs(f"./testOuts/{model}/fewShot")
    if selfConsistency:
        if noCoT:
            if not os.path.exists(f"./testOuts/{model}/fewShot/selfConsistency/noCoT"):
                os.makedirs(f"./testOuts/{model}/fewShot/selfConsistency/noCoT")
            if not os.path.exists(f"./testOuts/{model}/fewShot/selfConsistency/noCoT/{dataset}"):
                os.makedirs(f"./testOuts/{model}/fewShot/selfConsistency/noCoT/{dataset}")
        else:
            if not os.path.exists(f"./testOuts/{model}/fewShot/selfConsistency"):
                os.makedirs(f"./testOuts/{model}/fewShot/selfConsistency")
            if not os.path.exists(f"./testOuts/{model}/fewShot/selfConsistency/{dataset}"):
                os.makedirs(f"./testOuts/{model}/fewShot/selfConsistency/{dataset}")
    else: 
        if noCoT:
            if not os.path.exists(f"./testOuts/{model}/fewShot/noCoT/{dataset}"):
                os.makedirs(f"./testOuts/{model}/fewShot/noCoT/{dataset}")
        else:
            if not os.path.exists(f"./testOuts/{model}/fewShot/{dataset}"):
                os.makedirs(f"./testOuts/{model}/fewShot/{dataset}")

#Only do inferencing
with torch.no_grad():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    if selfConsistency:
        if model == "flant5":
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
        elif model =="mpt":
            raise NotImplementedError("Self consistency prompting not supported for MPT-7B!")
        elif model == "alpaca":
            raise NotImplementedError("Self consistency prompting not supported for Alpaca!")
        elif model == "llama":
            raise NotImplementedError("Self consistency prompting not supported for Llama!")
        else: 
            raise ValueError(f"{model} is not a reecognized model!")
    else:
        if model == "flant5":
            pipe_flan = transformers.pipeline(
                "text2text-generation", 
                model=f"google/flan-t5-{modelSize}", 
                device=device, 
                model_kwargs= {
                    "torch_dtype":torch.bfloat16
                }
            )
        elif model =="mpt":
            modelName =  "mosaicml/mpt-7b-instruct"
            modelConfig = transformers.AutoConfig.from_pretrained(
                modelName,
                trust_remote_code=True
            )
            modelConfig.attn_config['attn_impl'] = 'triton'
            modelConfig.update({"max_seq_len": 4096})
            modelMPT = transformers.AutoModelForCausalLM.from_pretrained(
                modelName,
                config=modelConfig,
                trust_remote_code=True
            )
            modelMPT.eval()
            modelMPT.to(device=device, dtype=torch.bfloat16)
            modelTokenizer = transformers.AutoTokenizer.from_pretrained(
                modelName, 
                trust_remote_code=True
            )
            if modelTokenizer.pad_token_id is None:
                modelTokenizer.pad_token = modelTokenizer.eos_token
            modelTokenizer.padding_side = "left"
            modelMPT_generate_kwargs = {
                "temperature": TEMPERATURE_MPT,
                "top_p": 0.92,
                "top_k": 0,
                "max_new_tokens": MAX_NEW_TOKENS_MPT,
                "use_cache": True,
                "do_sample": True,
                "eos_token_id": modelTokenizer.eos_token_id,
                "pad_token_id": modelTokenizer.pad_token_id,
                "repetition_penalty": 1.1,  # 1.0 means no penalty, > 1.0 means penalty, 1.2 from CTRL paper
            }
        elif model == "alpaca":
            pathToAlpaca = "/scratch/general/vast/u1266434/llama_models/alpaca-7b-recovered/"
            modelAlpaca = transformers.AutoModelForCausalLM.from_pretrained(
                pathToAlpaca,
            )
            modelTokenizer = transformers.AutoTokenizer.from_pretrained(
                pathToAlpaca,
            )
            special_tokens_dict = dict()
            if modelTokenizer.pad_token is None:
                special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
            if modelTokenizer.eos_token is None:
                special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
            if modelTokenizer.bos_token is None:
                special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
            if modelTokenizer.unk_token is None:
                special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=special_tokens_dict,
                tokenizer=modelTokenizer,
                model=modelAlpaca,
            )
            modelAlpaca.eval()
            modelAlpaca.to(device=device)
            # modelAlpaca_generate_kwargs = transformers.GenerationConfig(
            #     temperature=TEMPERATURE_ALPACA,
            #     top_p=0.92,
            #     top_k=0,
            #     # # temperature=0.8,
            #     # top_p=0.9,
            #     # top_k=40,
            #     # repetition_penalty=1.3,
            #     max_new_tokens=MAX_NEW_TOKENS_ALPACA,
            #     use_cache=True,
            #     do_sample=True,
            #     bos_token_id=modelTokenizer.bos_token_id,
            #     eos_token_id=modelTokenizer.eos_token_id,
            #     pad_token_id=modelTokenizer.pad_token_id,
            #     repetition_penalty=1.1
            # )
            modelAlpaca_generate_kwargs = {
                "temperature": TEMPERATURE_ALPACA,
                 "top_p": TOP_P_ALPACA,
                "top_k": TOP_K_ALPACA,
                "use_cache":USE_CACHE_ALPACA,
                "do_sample": DO_SAMPLE_ALPACA,
                "bos_token_id": modelTokenizer.bos_token_id,
                "eos_token_id": modelTokenizer.eos_token_id,
                "pad_token_id": modelTokenizer.pad_token_id,
                "repetition_penalty": REPETITION_PENALTY_ALPACA,
            }
            dsModel = deepspeed.init_inference(
                model=modelAlpaca,
                # dtype=torch.half,
                mp_size=1,
                replace_method="auto",
                replace_with_kernel_inject=True,
                max_out_tokens=MAX_LENGTH_ALPACA,
            )
        elif model == "llama":
            if modelSize == "13b":
                pathToLlama = "huggyllama/llama-13b"
            else:
                pathToLlama = "/scratch/general/vast/u1266434/llama_models/hf_7b"
            modelLlama = LlamaForCausalLM.from_pretrained(pathToLlama)
            modelTokenizer = LlamaTokenizer.from_pretrained(pathToLlama)
            special_tokens_dict = dict()
            if modelTokenizer.pad_token is None:
                special_tokens_dict["pad_token"] = DEFAULT_PAD_TOKEN
            if modelTokenizer.eos_token is None:
                special_tokens_dict["eos_token"] = DEFAULT_EOS_TOKEN
            if modelTokenizer.bos_token is None:
                special_tokens_dict["bos_token"] = DEFAULT_BOS_TOKEN
            if modelTokenizer.unk_token is None:
                special_tokens_dict["unk_token"] = DEFAULT_UNK_TOKEN

            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=special_tokens_dict,
                tokenizer=modelTokenizer,
                model=modelLlama,
            )
            modelLlama.eval()
            modelLlama.to(device=device)
            # modelLlama_generate_kwargs = transformers.GenerationConfig(
            #     temperature=TEMPERATURE_LLAMA,
            #     top_p=0.92,
            #     top_k=0,
            #     max_new_tokens=MAX_NEW_TOKENS_LLAMA,
            #     use_cache=True,
            #     do_sample=True,
            #     bos_token_id=modelTokenizer.bos_token_id,
            #     eos_token_id=modelTokenizer.eos_token_id,
            #     pad_token_id=modelTokenizer.pad_token_id,
            #     repetition_penalty=1.1
            # )
            modelLlama_generate_kwargs = {
                "temperature": TEMPERATURE_LLAMA,
                "top_p": TOP_P_LLAMA,
                "top_k": TOP_K_LLAMA,
                "use_cache":USE_CACHE_LLAMA,
                "do_sample": DO_SAMPLE_LLAMA,
                "bos_token_id": modelTokenizer.bos_token_id,
                "eos_token_id": modelTokenizer.eos_token_id,
                "pad_token_id": modelTokenizer.pad_token_id,
                "repetition_penalty": REPETITION_PENALTY_LLAMA,
            }
            # modelLlama_generate_kwargs = {
            #     "use_cache":True,
            #     "do_sample": False,
            #     "bos_token_id": modelTokenizer.bos_token_id,
            #     "eos_token_id": modelTokenizer.eos_token_id,
            #     "pad_token_id": modelTokenizer.pad_token_id,
            # }
            dsModel = deepspeed.init_inference(
                model=modelLlama,
                # dtype=torch.half,
                # mp_size=1,
                # replace_method="auto",
                # replace_with_kernel_inject=True,
                max_out_tokens=MAX_LENGTH_LLAMA,
            )
        else: 
            raise ValueError(f"{model} is not a reecognized model!")
    if model == "flant5":
        logging.info(f"Model: FLANT5-{modelSize}")
    elif model =="mpt":
        logging.info("Model: MPT-7B")
    elif model =="alpaca":
        logging.info("Model: Alpaca-7B")
    elif model =="llama":
        logging.info("Model: Llama-{}".format(modelSize))
    else: 
        raise ValueError(f"{model} is not a reecognized model!")
    if selfConsistency:
        logging.info("Decoding: Self Consistency")
    else:
        logging.info("Decoding: Greedy")
    if noCoT:
        logging.info("Chain of Thought Prompting: False")
    else: 
        logging.info("Chain of Thought Prompting: True")
    if zeroShot:
        logging.info("Prompting: Zero Shot")
    else:
        logging.info("Prompting: Few Shot")
        logging.info("Max Shots: {}".format(maxShots))
    logging.info("Dataset: {}".format(dataset))

    if dataset in huggingfaceDatasets:
        datasetHF = load_dataset(huggingfaceDatasetsNames[dataset])
        if dataset == "boolqHF":
            datasetHF = datasetHF.rename_column("passage","sentence1")
            datasetHF = datasetHF.rename_column("question","sentence2")
            datasetHF = datasetHF.rename_column("answer","label")
        elif dataset == "condaqaHF":
            pass
        elif dataset == "quorefHF":
            def quorefHFmap(example):
                example["answer"] = example["answers"]["text"][0]
                return example
            for split in datasetHF:
                if "answers" in datasetHF[split].column_names:
                    datasetHF[split] = datasetHF[split].map(quorefHFmap)
                    datasetHF[split] = datasetHF[split].remove_columns(["answers"])
            datasetHF = datasetHF.rename_column("context","sentence1")
            datasetHF = datasetHF.rename_column("question","sentence2")
            datasetHF = datasetHF.rename_column("answer","label")
        elif dataset == "ropesHF":
            def ropesHFmap(example):
                example["context"] = example["background"] + " " + example["situation"] 
                example["answer"] = example["answers"]["text"][0]
                return example
            for split in datasetHF:
                if "answers" in datasetHF[split].column_names:
                    datasetHF[split] = datasetHF[split].map(quorefHFmap)
                    datasetHF[split] = datasetHF[split].remove_columns(["answers"])
            datasetHF = datasetHF.rename_column("context","sentence1")
            datasetHF = datasetHF.rename_column("question","sentence2")
            datasetHF = datasetHF.rename_column("answer","label")
        else: 
            raise ValueError(f"{dataset} not supported!")

    for trainFile in tqdm(trainFiles,desc="Train File"):
        #Contingency
        #Remove after first successful run
        logging.info(f"#{trainFile}")
        #---------------------------------
        if not zeroShot:
            if dataset not in huggingfaceDatasets or trainFile.endswith(".json"):
                trainData = readJSON(trainFile, dataset)
            else:
                trainData = list(datasetHF[trainFile].select(np.random.choice(len(datasetHF[trainFile]), maxShots)))
            trainPrompt = generateTrainPrompt(trainData, promptType, dataset, noCoT, model, maxShots, bestPromptType)
        for testFile in tqdm(testFiles,desc="Test File"):
            #Contingency
            #Remove after first successful run
            logging.info(f"##{testFile}")
            #---------------------------------
            if dataset not in huggingfaceDatasets or testFile.endswith(".json"):
                testData = readJSON(testFile, dataset)
            else:
                testData = list(datasetHF[testFile])
            testOuts = []
            testInd = -1
            avgShotLen = 0
            for testEx in tqdm(testData,desc="Test Instance"):
                testInd +=1
                testPrompt = generateTestPrompt(testEx, promptType, dataset, noCoT, model, maxShots, bestPromptType)
                if not zeroShot:
                    finalPrompt = ("".join(trainPrompt)) + ("".join(testPrompt))
                else:
                    finalPrompt = ("".join(testPrompt))                 

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if selfConsistency:
                    if model == "flant5":
                        output_flan = pipe_flan(finalPrompt, max_length=MAX_LENGTH)

                        sampleID = f"{trainFile}_{testFile}_{testInd}"

                        for out in output_flan: 
                            newEx = testEx.copy()
                            newEx["SampleID"] = sampleID
                            newEx["output"] = out["generated_text"]
                            testOuts.append(newEx)
                    elif model =="mpt":
                        raise NotImplementedError("Self-consistency prompting not implemented for MPT-7B model!")
                    elif model =="alpaca":
                        raise NotImplementedError("Self-consistency prompting not implemented for Alpaca model!")
                    elif model =="llama":
                        raise NotImplementedError("Self-consistency prompting not implemented for Llama model!")
                    else: 
                        raise ValueError(f"{model} is not a recognized model!")
                else:
                    if model == "flant5":
                        output_flan = pipe_flan(finalPrompt, max_length=MAX_LENGTH)[0]["generated_text"]
                    elif model == "mpt":
                        modelOuputs = modelTokenizer(finalPrompt, return_tensors="pt")
                        modelInputIDs = modelOuputs.input_ids
                        modelInputIDs = modelInputIDs.to(device)
                        modelAttnMasks = modelOuputs.attention_mask
                        modelAttnMasks = modelAttnMasks.to(device)
                        outputIDs = modelMPT.generate(modelInputIDs, **modelMPT_generate_kwargs, attention_mask=modelAttnMasks)

                        #Causal LM is used here
                        newTokens = outputIDs[0, len(modelInputIDs[0]):]
                        output_flan = modelTokenizer.decode(newTokens, skip_special_tokens=True)
                        logging.info(finalPrompt)
                        logging.info(output_flan)
                        logging.info("*"*20)
                    elif model =="alpaca":
                        modelOuputs = modelTokenizer(finalPrompt, return_tensors="pt")
                        modelInputIDs = modelOuputs.input_ids
                        modelInputIDs = modelInputIDs.to(device)
                        modelAttentionMask = modelOuputs.attention_mask
                        modelAttentionMask = modelAttentionMask.to(device)
                        # outputIDs = modelAlpaca.generate(
                        #     input_ids=modelInputIDs,
                        #     generation_config=modelAlpaca_generate_kwargs,
                        # )

                        outputIDs = dsModel.module.generate(
                            input_ids=modelInputIDs,
                            attention_mask=modelAttentionMask,
                            max_new_tokens=MAX_NEW_TOKENS_ALPACA,
                            **modelAlpaca_generate_kwargs,
                        )

                        #Causal LM is used here
                        newTokens = outputIDs[0, len(modelInputIDs[0]):]
                        output_flan = modelTokenizer.decode(newTokens, skip_special_tokens=True)
                        logging.info(finalPrompt)
                        logging.info(output_flan)
                        logging.info("*"*20)
                    elif model =="llama":
                        #LLama models cannot handle more than 2048 tokens in context
                        hitLowerLimit = False
                        curMaxShots = maxShots
                        while(1):
                            if curMaxShots < 0:
                                logging.error("Could not fit an instance within context window!")
                                hitLowerLimit = True
                                break
                            modelOuputs = modelTokenizer(finalPrompt, return_tensors="pt")
                            modelInputIDs = modelOuputs.input_ids
                            if modelInputIDs.shape[-1] <= (MAX_CONTEXT_LENGTH_LLAMA-MAX_NEW_TOKENS_LLAMA):
                                avgShotLen += curMaxShots
                                break
                            curMaxShots -= 1
                            if curMaxShots > 0:
                                trainPrompt = generateTrainPrompt(trainData, promptType, dataset, noCoT, model, curMaxShots, bestPromptType)
                            if not zeroShot and curMaxShots > 0:
                                finalPrompt = ("".join(trainPrompt)) + ("".join(testPrompt))
                            else:
                                finalPrompt = ("".join(testPrompt))
                        if hitLowerLimit:
                            continue
                        modelAttentionMask = modelOuputs.attention_mask
                        modelInputIDs = modelInputIDs.to(device)
                        modelAttentionMask = modelAttentionMask.to(device)
                        # outputIDs = modelLlama.generate(
                        #     input_ids=modelInputIDs,
                        #     attention_mask=modelAttentionMask,
                        #     generation_config=modelLlama_generate_kwargs,
                        # )
                        #Only deepspeed inference
                        outputIDs = dsModel.module.generate(
                            input_ids=modelInputIDs,
                            attention_mask=modelAttentionMask,
                            max_new_tokens=MAX_NEW_TOKENS_LLAMA,
                            **modelLlama_generate_kwargs,
                        )

                        #Causal LM is used here
                        newTokens = outputIDs[0, len(modelInputIDs[0]):]
                        output_flan = modelTokenizer.decode(newTokens, skip_special_tokens=True)
                        logging.info(finalPrompt)
                        logging.info(output_flan)
                        logging.info("*"*20)
                    else: 
                        raise ValueError(f"{model} is not a reecognized model!")

                    testEx["output"] = output_flan
                    testOuts.append(testEx)                           

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if not zeroShot and model =="llama":
                avgShotLen /= len(testData)
                logging.info("Average no. of few shot exemplars in prompt: {}".format(avgShotLen))

            if zeroShot:
                if selfConsistency:
                    if noCoT:
                        filePath = f'./testOuts/{model}/zeroShot/selfConsistency/noCoT/{dataset}/{trainFile.split("/")[-1].split(".")[0]}__{testFile.split("/")[-1].split(".")[0]}__{promptType}__{zeroShot}__{model}.json'
                    else: 
                        filePath = f'./testOuts/{model}/zeroShot/selfConsistency/{dataset}/{trainFile.split("/")[-1].split(".")[0]}__{testFile.split("/")[-1].split(".")[0]}__{promptType}__{zeroShot}__{model}.json'
                else: 
                    if noCoT:
                        filePath = f'./testOuts/{model}/zeroShot/noCoT/{dataset}/{trainFile.split("/")[-1].split(".")[0]}__{testFile.split("/")[-1].split(".")[0]}__{promptType}__{zeroShot}__{model}.json'
                    else: 
                        filePath = f'./testOuts/{model}/zeroShot/{dataset}/{trainFile.split("/")[-1].split(".")[0]}__{testFile.split("/")[-1].split(".")[0]}__{promptType}__{zeroShot}__{model}.json'
            else:
                if selfConsistency:
                    if noCoT:
                        filePath = f'./testOuts/{model}/fewShot/selfConsistency/noCoT/{dataset}/{trainFile.split("/")[-1].split(".")[0]}__{testFile.split("/")[-1].split(".")[0]}__{promptType}__{zeroShot}__{model}.json'
                    else: 
                        filePath = f'./testOuts/{model}/fewShot/selfConsistency/{dataset}/{trainFile.split("/")[-1].split(".")[0]}__{testFile.split("/")[-1].split(".")[0]}__{promptType}__{zeroShot}__{model}.json'
                else: 
                    if noCoT:
                        filePath = f'./testOuts/{model}/fewShot/noCoT/{dataset}/{trainFile.split("/")[-1].split(".")[0]}__{testFile.split("/")[-1].split(".")[0]}__{promptType}__{zeroShot}__{model}.json'
                    else: 
                        filePath = f'./testOuts/{model}/fewShot/{dataset}/{trainFile.split("/")[-1].split(".")[0]}__{testFile.split("/")[-1].split(".")[0]}__{promptType}__{zeroShot}__{model}.json'
            
            with open(filePath, 'w') as fout:
                json.dump(testOuts , fout)
        logging.info("*****")
    logging.info("-----")
