import json
from os.path import exists
from pathlib import Path
import argparse
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
    "-model",
    choices=["flant5", "mpt", "alpaca"],
    help="Name of model to use for inference",
    default="flant5"
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
noCoT = args.noCoT
model = args.model
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
        "llama2": {
            #Default Case
            -1: {
                "default": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n",
                "isTest": "",
                # "boolq": {
                #     "default": "Answer the questions with yes/no/don't know\n",
                #     "isTest": ""
                # }
            },
        },
#          "llama2": {
#             #Default Case
#             -1: {
#                 "default":"""<s>[INST] <<SYS>>
# Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
# <</SYS>>

# """,
#                 "isTest": "",
#             },
#         }
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
        "llama2": {
            1: {
                "CoT": {
                    "default": "### Passage:\n{passage}\n### Question:\n{question}?\n### Response:\n### Explanation:\n{explanation}\n### Answer:\n{answer}\n\n",
                    "imdb": "### Review:\n{review}\n### Question:\n{question}\n### Response:\n### Explanation:\n{explanation}\n### Answer:\n{answer}\n\n",
                    "perspectrum": "### Perspective:\n{perspective}\n### Claim:\n{claim}\n### Question:\n{question}\n### Instruction:\n{instruction}\n### Response:\n### Explanation:\n{explanation}\n### Answer:\n{answer}\n\n",
                },
                "noCoT": {
                    "default": "### Passage:\n{passage}\n### Question:\n{question}?\n### Answer:\n{answer}\n\n",
                    "imdb": "### Review:\n{review}\n### Question:\n{question}\n### Answer:\n{answer}\n\n",
                    "perspectrum": "### Perspective:\n{perspective}\n### Claim:\n{claim}\n### Question:\n{question}\n### Instruction:\n{instruction}\n### Answer:\n{answer}\n\n",
                    "udparsing": "### Sentence:\n{sentence}\n### Question:\n{question}\n### Answer:\n{answer}\n\n",
                },
                "isTest": {
                    "CoT": {
                        "default": "### Passage:\n{passage}\n### Question:\n{question}?\n### Response:\n",
                        "imdb": "### Review:\n{review}\n### Question:\n{question}\n### Response:\n",
                        "perspectrum": "### Perspective:\n{perspective}\n### Claim:\n{claim}\n### Question:\n{question}\n### Instruction:\n{instruction}\n### Response:\n",
                    },
                    "noCoT": {
                        "default": "### Passage:\n{passage}\n### Question:\n{question}?\n### Answer:\n",
                        "imdb": "### Review:\n{review}\n### Question:\n{question}\n### Answer:\n",
                        "perspectrum": "### Perspective:\n{perspective}\n### Claim:\n{claim}\n### Question:\n{question}\n### Instruction:\n{instruction}\n### Answer:\n",
                        "udparsing": "### Sentence:\n{sentence}\n### Question:\n{question}\n### Answer:\n",
                    },
                }
            }, 
        }, 
        # "llama2": {
        #     1: {
        #         "CoT": {
        #             "default": "{passage} {question}?[/INST] {explanation} Therefore, the answer is {answer}.</s>",
        #             # "default": "Passage:\n{passage}\nQuestion:\n{question}?[/INST] {explanation} Therefore, the answer is {answer}.</s>",
        #             # "imdb": "### Review:\n{review}\n### Question:\n{question}\n### Response:\n### Explanation:\n{explanation}\n### Correct Answer:\n{answer}\n\n",
        #             # "perspectrum": "### Perspective:\n{perspective}\n### Claim:\n{claim}\n### Question:\n{question}\n### Instruction:\n{instruction}\n### Response:\n### Explanation:\n{explanation}\n### Correct Answer:\n{answer}\n\n",
        #         },
        #         "noCoT": {
        #             "default": "{passage} {question}?[/INST] The answer is {answer}.</s>",
        #             # "default": "Passage:\n{passage}\nQuestion:\n{question}?[/INST] The answer is {answer}.</s>",
        #             # "imdb": "### Review:\n{review}\n### Question:\n{question}\n### Response:\n{answer}\n\n",
        #             # "perspectrum": "### Perspective:\n{perspective}\n### Claim:\n{claim}\n### Question:\n{question}\n### Instruction:\n{instruction}\n### Response:\n{answer}\n\n",
        #             # "udparsing": "### Sentence:\n{sentence}\n### Question:\n{question}\n### Response:\n{answer}\n\n",
        #         },
        #         "isTest": {
        #             "CoT": {
        #                 "default": "{passage} {question}?[/INST]",
        #                 # "default": "Passage:\n{passage}\nQuestion:\n{question}?[/INST]",
        #                 # "imdb": "### Review:\n{review}\n### Question:\n{question}\n### Response:\n",
        #                 # "perspectrum": "### Perspective:\n{perspective}\n### Claim:\n{claim}\n### Question:\n{question}\n### Instruction:\n{instruction}\n### Response:\n",
        #             },
        #             "noCoT": {
        #                 "default": "{passage} {question}?[/INST]",
        #                 # "default": "Passage:\n{passage}\nQuestion:\n{question}?[/INST]",
        #                 # "imdb": "### Review:\n{review}\n### Question:\n{question}\n### Response:\n",
        #                 # "perspectrum": "### Perspective:\n{perspective}\n### Claim:\n{claim}\n### Question:\n{question}\n### Instruction:\n{instruction}\n### Response:\n",
        #                 # "udparsing": "### Sentence:\n{sentence}\n### Question:\n{question}\n### Response:\n",
        #             },
        #         }
        #     }, 
        # },
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
def generateTrainPrompt(data, promptType, dataset, noCoT, model, bestPromptType=1):
    return _generatePrompt(data,promptType, dataset, noCoT, model, bestPromptType)
#---------------------------------------------------------------------------
def generateTestPrompt(data, promptType, dataset, noCoT, model, bestPromptType=1):
    return _generatePrompt([data],promptType, dataset, noCoT, model, bestPromptType, True)[0]
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

for trainFile in trainFiles:
    #Contingency
    #Remove after first successful run
    logging.info(f"#{trainFile}")
    #---------------------------------
    if not zeroShot:
        trainData = readJSON(trainFile, dataset)
        trainPrompt = generateTrainPrompt(trainData, promptType, dataset, noCoT, model, bestPromptType)
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
            testPrompt = generateTestPrompt(testEx, promptType, dataset, noCoT, model, bestPromptType)
            if not zeroShot:
                finalPrompt = ("".join(trainPrompt)) + testPrompt
            else:
                finalPrompt = testPrompt                        
            print(finalPrompt)
    logging.info("*****")
logging.info("-----")
