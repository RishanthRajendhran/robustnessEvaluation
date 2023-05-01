import json
import argparse
import numpy as np
import os
import glob
from nltk import tokenize 
from transformers import BertTokenizer
import string
from unidecode import unidecode
import regex as re

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def checkAnswers(answer, corrAnswer, tokenizer=None):

    #Remove accents from text and lowercase all characters
    answer = unidecode(answer.lower())
    corrAnswer = unidecode(corrAnswer.lower())
    answer = "".join([s for s in answer if s not in string.punctuation])
    corrAnswer = "".join([s for s in corrAnswer if s not in string.punctuation])

    if answer == corrAnswer:
        em = 1
    else:
        em = 0

    wordsInAnswer = tokenize.word_tokenize(answer)
    wordsInCorrAnswer = tokenize.word_tokenize(corrAnswer) 
    # if tokenizer == None:
        # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # wordsInAnswer = tokenizer.tokenize(answer)
    # wordsInCorrAnswer = tokenizer.tokenize(corrAnswer)  
    
    # commonWords = wordsInCorrAnswer.intersection(wordsInAnswer)
    commonWords = np.intersect1d(wordsInCorrAnswer, wordsInAnswer)
    if len(wordsInAnswer) == 0:
        precision = 0
    else:
        precision = len(commonWords)/len(wordsInAnswer)
    if len(wordsInCorrAnswer) == 0:
        recall = 0
    else:
        recall = len(commonWords)/len(wordsInCorrAnswer)
    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = (2*precision*recall)/(precision+recall)
    # if f1<1:
    #     print(f"F1 score      : {f1}")
    #     print(f"Prediction    : {answer}")
    #     print(f"Correct Answer: {corrAnswer}")
    #     print("********")
    
    return f1, em

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
    "-concise",
    action="store_true",
    help="Boolean flag to indicate if outputs need to be concise"
)

parser.add_argument(
    "-summaryOnly",
    action="store_true",
    help="Booleaan flag to indicate if only summary needs to be printed"
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
        "udparsing"
    ],
    required=True,
)

parser.add_argument(
    "-selfConsistency",
    action="store_true",
    help="Booleaan flag to compute self-consistency"
)

parser.add_argument(
    "-f1",
    "--f1Threshold",
    type=float,
    help="F1 Threshold to use for evaluation (between 0 and 1 only)",
    default=0.8
)

parser.add_argument(
    "-pattern",
    help="RegEx pattern for json file names in the output directory that need to be evaluated",
    default=None
)

args = parser.parse_args()
outputFiles = args.outputFiles
isDirectory = args.isDirectory
concise = args.concise
summaryOnly = args.summaryOnly
dataset = args.dataset
selfConsistency = args.selfConsistency
f1Threshold = args.f1Threshold
pattern = args.pattern
if f1Threshold < 0 or f1Threshold > 1:
    raise ValueError(f"F1 Threshold should be between 0 and 1")

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
#---------------------------------------------------------------------------
def extractAnswer(answer, isOut):
    old = answer
    answer = answer.strip()
    answer = answer.lower()
    if answer[-1] == ".":
        answer = answer[:-1]
    if isOut:
        if "answer is " in answer:
            answer = answer[-(answer[-1::-1].index("answer is "[-1::-1])):]
            # answer = answer[answer.index("answer is ")+len("answer is "):]
        elif "answer " in answer:
            answer = answer[-(answer[-1::-1].index("answer "[-1::-1])):]
            # answer = answer[answer.index("answer ")+len("answer "):]
        answer = answer.strip()
        if answer[-1] == ".":
            answer = answer[:-1]
    #For BoolQ 
    if answer.lower() == "true":
        answer = "yes"
    elif answer.lower() == "false":
        answer = "no"
    #For perspectrum
    elif answer.lower() == "supports":
        answer = "pos"
    elif answer.lower() == "undermines":
        answer = "neg"
    elif answer.lower() == "not a valid perspective":
        answer = "unk"
    return answer
#---------------------------------------------------------------------------
def main():
    promptPerfs = {}
    outputFiles.sort()
    for outputFile in outputFiles:
        passages = {}
        if selfConsistency:
            selfPassages = {}
        data = []   
        numEditTypes = {}
        for  line in open(outputFile, "r"):
            data.extend(json.loads(line))
        # with open(outputFile, "r") as f:
        #     data = json.load(f)
        for d in data:
            if len(d["output"]) == 0 or len(d["label"]) == 0:
                continue
            answer = d["output"]
            answer = extractAnswer(answer, True)
            corrAnswer = d["label"]
            corrAnswer = extractAnswer(corrAnswer, False)
            passId = d["PassageID"]
            questionID = d["QuestionID"]
            isOriginal = d["isOriginal"]
            if selfConsistency:
                if "SampleID" in d.keys():
                    sampleID = d["SampleID"]
                else: 
                    sampleID = d["sampleID"]
                if sampleID not in selfPassages.keys():
                    selfPassages[sampleID] = {
                            "passageID": passId,
                            "questionID": questionID,
                            "isOriginal": isOriginal,
                            "answers": [],
                            "label": corrAnswer,
                            "score": 0,
                            "count": 1,
                            "failureCases": [],
                        }    
                selfPassages[sampleID]["answers"].append(answer) 
            else:
                if dataset == "condaqa":
                    if d["PassageEditID"] not in numEditTypes.keys():
                        numEditTypes[d["PassageEditID"]] = 1
                    else:
                        numEditTypes[d["PassageEditID"]] += 1
                if passId not in passages.keys():
                    if dataset in ["condaqa", "perspectrum", "ropes", "quoref", "drop"]:
                        passages[passId] = {}
                    else:
                        passages[passId] = {
                            "score": 0,
                            "count": 0,
                            "f1": 0,
                            "f1Contrast": 0,
                            "em": 0,
                            "emContrast": 0,
                            "scoreContrast": 0,
                            "countContrast": 0,
                            "countOriginal": 0,
                            "failureEditTypes": [],
                            "failureCases": [],
                        }
                if dataset in ["condaqa", "perspectrum", "ropes", "quoref", "drop"]:
                    if questionID not in passages[passId].keys():
                        passages[passId][questionID] = {
                            "score": 0,
                            "count": 0,
                            "f1": 0,
                            "f1Contrast": 0,
                            "em": 0,
                            "emContrast": 0,
                            "scoreContrast": 0,
                            "countContrast": 0,
                            "countOriginal": 0,
                            "failureEditTypes": [],
                            "failureCases": [],
                        }
                if dataset in ["condaqa", "perspectrum", "ropes", "quoref", "drop"]:
                    passages[passId][questionID]["count"] += 1
                    if not isOriginal:
                        passages[passId][questionID]["countContrast"] += 1
                    else: 
                        passages[passId][questionID]["countOriginal"] += 1
                else:
                    passages[passId]["count"] += 1
                    if not isOriginal:
                        passages[passId]["countContrast"] += 1
                    else: 
                        passages[passId]["countOriginal"] += 1

                # if answer == corrAnswer or ((" " + corrAnswer + " ") in answer and abs(len(corrAnswer)-len(answer))<=6) or ((" " + answer + " ") in  corrAnswer and abs(len(corrAnswer)-len(answer))<=6):
                f1, em = checkAnswers(answer, corrAnswer, tokenizer)
                #Update EM Scores
                if dataset in ["condaqa", "perspectrum", "ropes", "quoref", "drop"]:
                    passages[passId][questionID]["em"] += em
                    if not isOriginal:
                        passages[passId][questionID]["emContrast"] += em
                    else: 
                        passages[passId][questionID]["emContrast"] += 0
                else: 
                    passages[passId]["em"] += em
                    if not isOriginal:
                        passages[passId]["emContrast"] += em
                    else: 
                        passages[passId]["emContrast"] += 0
                #Update F1 Scores
                if f1 >= f1Threshold:
                    if dataset in ["condaqa", "perspectrum", "ropes", "quoref", "drop"]:
                        passages[passId][questionID]["score"] += 1
                        passages[passId][questionID]["f1"] += f1
                        if not isOriginal:
                            passages[passId][questionID]["f1Contrast"] += f1
                            passages[passId][questionID]["scoreContrast"] += 1
                        else: 
                            passages[passId][questionID]["f1Contrast"] += 0
                    else: 
                        passages[passId]["score"] += 1
                        passages[passId]["f1"] += f1
                        if not isOriginal:
                            passages[passId]["f1Contrast"] += f1
                            passages[passId]["scoreContrast"] += 1
                        else: 
                            passages[passId]["f1Contrast"] += 0
                else:
                    if dataset in ["condaqa", "perspectrum", "ropes", "quoref", "drop"]:
                        passages[passId][questionID]["f1"] += f1
                        if not isOriginal:
                            passages[passId][questionID]["f1Contrast"] += f1
                        else: 
                            passages[passId][questionID]["f1Contrast"] += 0
                        if dataset == "condaqa":
                            passages[passId][questionID]["failureEditTypes"].append(d["PassageEditID"])
                        passages[passId][questionID]["failureCases"].append({"output":answer, "label":corrAnswer})
                    else:
                        passages[passId]["f1"] += f1
                        if not isOriginal:
                            passages[passId]["f1Contrast"] += f1
                        else: 
                            passages[passId]["f1Contrast"] += 0
                        passages[passId]["failureCases"].append({"output":answer, "label":corrAnswer})
        if selfConsistency:
            for sampleID in selfPassages.keys():
                if len(selfPassages[sampleID]["answers"]) != 10:
                    print("{}: {}".format(sampleID, len(selfPassages[sampleID]["answers"])))
                values, counts = np.unique(selfPassages[sampleID]["answers"], return_counts=True)
                majorityOutput = values[np.argmax(counts)]
                answer = majorityOutput
                corrAnswer = selfPassages[sampleID]["label"]

                passId = selfPassages[sampleID]["passageID"]
                questionID = selfPassages[sampleID]["questionID"]
                if dataset in ["condaqa", "quoref", "ropes", "drop", "perspectrum"]:
                    if passId not in passages.keys():
                        passages[passId] = {}
                    if questionID not in passages[passId].keys():
                            passages[passId][questionID] = {
                                "score": 0,
                                "count": 0,
                                "f1": 0,
                                "f1Contrast": 0,
                                "em": 0,
                                "emContrast": 0,
                                "scoreContrast": 0,
                                "countContrast": 0,
                                "countOriginal": 0,
                                "failureEditTypes": [],
                                "failureCases": [],
                        }
                    passages[passId][questionID]["count"] += 1
                    if not selfPassages[sampleID]["isOriginal"]:
                        passages[passId][questionID]["countContrast"] += 1
                    else:
                        passages[passId][questionID]["countOriginal"] += 1
                    f1, em = checkAnswers(answer, corrAnswer, tokenizer)
                    #Update EM Score
                    passages[passId][questionID]["em"] += em
                    if not selfPassages[sampleID]["isOriginal"]:
                        passages[passId][questionID]["emContrast"] += em 
                    else: 
                        passages[passId][questionID]["emContrast"] += 0
                    #Update F1 Score
                    passages[passId][questionID]["f1"] += f1
                    if not selfPassages[sampleID]["isOriginal"]:
                        passages[passId][questionID]["f1Contrast"] += f1
                    else:
                        passages[passId][questionID]["f1Contrast"] += 0
                    if f1 >= f1Threshold:
                        passages[passId][questionID]["score"] += 1
                        if not selfPassages[sampleID]["isOriginal"]:
                            passages[passId][questionID]["scoreContrast"] += 1
                    else:
                        passages[passId][questionID]["failureCases"].append({"output":answer, "label":corrAnswer})
                elif dataset in ["mctaco", "boolq", "imdb", "matres"]:
                    if passId not in passages.keys():
                        passages[passId] = {
                                "score": 0,
                                "count": 0,
                                "f1": 0,
                                "f1Contrast": 0,
                                "em": 0,
                                "emContrast": 0,
                                "scoreContrast": 0,
                                "countContrast": 0,
                                "countOriginal": 0,
                                "failureEditTypes": [],
                                "failureCases": [],
                        }
                    passages[passId]["count"] += 1
                    if not selfPassages[sampleID]["isOriginal"]:
                        passages[passId]["countContrast"] += 1
                    else: 
                        passages[passId]["countOriginal"] += 1
                    f1, em = checkAnswers(answer, corrAnswer, tokenizer)
                    #Update EM Score
                    passages[passId]["em"] += em
                    if not selfPassages[sampleID]["isOriginal"]:
                        passages[passId]["emContrast"] += em 
                    else: 
                        passages[passId]["emContrast"] += 0
                    #Update F1 Score
                    passages[passId]["f1"] += f1
                    if not selfPassages[sampleID]["isOriginal"]:
                        passages[passId]["f1Contrast"] += f1
                    else:
                        passages[passId]["f1Contrast"] += 0
                    if f1 >= f1Threshold:
                        passages[passId]["score"] += 1
                        if not selfPassages[sampleID]["isOriginal"]:
                            passages[passId]["scoreContrast"] += 1
                    else:
                        passages[passId]["failureCases"].append({"output":answer, "label":corrAnswer})
                else:
                    raise RuntimeError("Self consistency only supported for CondaQA/BoolQ/MCTACO/Quoref/Ropes/MATRES/UDParsing datasets as of now!")
        accuracy = 0
        accuracyContrast = 0
        consistency = 0
        f1Scores = 0
        f1ScoresContrast = 0
        emScores = 0
        emScoresContrast = 0
        totalCounts = 0
        totalCountsContrast = 0
        numQues = 0
        failureEditTypes = []
        failureCases = []
        numMisses = {}
        for pID in passages.keys():
            if dataset in ["condaqa", "perspectrum", "ropes", "quoref", "drop"]:
                # Ignore contrast sets with only one question (probably the case that 
                # original question has no perturbed version)
                numQues += len(passages[pID].keys())
                for qID in passages[pID].keys():
                    accuracy += passages[pID][qID]["score"]
                    totalCounts += passages[pID][qID]["count"]
                    accuracyContrast += passages[pID][qID]["scoreContrast"]
                    totalCountsContrast += passages[pID][qID]["countContrast"]
                    f1Scores += passages[pID][qID]["f1"]
                    f1ScoresContrast += passages[pID][qID]["f1Contrast"]
                    emScores += passages[pID][qID]["em"]
                    emScoresContrast += passages[pID][qID]["emContrast"]
                    if passages[pID][qID]["count"] == 1:
                        numQues -= 1
                    else:
                        if passages[pID][qID]["score"] == passages[pID][qID]["count"]:
                            consistency += 1
                        if (passages[pID][qID]["count"]-passages[pID][qID]["score"]) not in numMisses.keys():
                            numMisses[(passages[pID][qID]["count"]-passages[pID][qID]["score"])] = 0
                        numMisses[(passages[pID][qID]["count"]-passages[pID][qID]["score"])] += 1
                    if dataset == "condaqa" and (not selfConsistency):
                        failureEditTypes.extend(passages[pID][qID]["failureEditTypes"])
                    failureCases.extend(passages[pID][qID]["failureCases"])
            else:
                # Ignore contrast sets with only one question (probably the case that 
                # original question has no perturbed version)
                
                if passages[pID]["count"] > 1:
                    # if passages[pID]["countOriginal"] > 1:
                    #     print(pID)
                    #     print(passages[pID])
                    #     exit(0)
                    numQues += 1
                    accuracy += passages[pID]["score"]
                    totalCounts += passages[pID]["count"]
                    accuracyContrast += passages[pID]["scoreContrast"]
                    totalCountsContrast += passages[pID]["countContrast"]
                    f1Scores += passages[pID]["f1"]
                    f1ScoresContrast += passages[pID]["f1Contrast"]
                    emScores += passages[pID]["em"]
                    emScoresContrast += passages[pID]["emContrast"]
                    if passages[pID]["score"] == passages[pID]["count"]:
                        consistency += 1
                    if (passages[pID]["count"]-passages[pID]["score"]) not in numMisses.keys():
                        numMisses[(passages[pID]["count"]-passages[pID]["score"])] = 0
                    numMisses[(passages[pID]["count"]-passages[pID]["score"])] += 1
                # else:
                #     print(pID)
                failureCases.extend(passages[pID]["failureCases"]) 
        accuracy /= totalCounts
        accuracyContrast /= totalCountsContrast
        consistency /= numQues
        f1Scores /= totalCounts
        f1ScoresContrast /= totalCountsContrast
        emScores /= totalCounts
        emScoresContrast /= totalCountsContrast
        numMisses = dict(sorted(numMisses.items()))

        if not summaryOnly:
            print(f"File: {outputFile}")
            print(f"\tNo. of examples: {totalCounts}")
            print(f"\tNo. of examples (w/o original examples): {totalCountsContrast}")
            print(f"\tNo. of contrast sets: {numQues}")
        parts = outputFile.split("__")
        curPromptType = None
        if len(parts) == 4:
            trainSet = parts[0].split("_")[-1]
            if not summaryOnly:
                print(f"Train file: {parts[0]}.json")
                print(f"Test file: {parts[1]}.json")
                print(f"Prompt type: {parts[2]}")
            curPromptType = parts[2]
            parts[3] = parts[3].split(".")[0]
            if not summaryOnly:
                print(f"Zero Shot?: {parts[3]}")
        else:
            print(f"Unable to retrieve statistics from input file names!")
        if not summaryOnly:
            print(f"\tAccuracy: {round(accuracy*100,2):0.2f}%")
            print(f"\tAccuracy (w/o original questions): {round(accuracyContrast*100, 2):0.2f}%")
            print(f"\tConsistency: {round(consistency*100,2):0.2f}%")
            print(f"\tF1 Score: {f1Scores*100:0.2f}%")
            print(f"\tF1 Score (w/o original questions): {f1ScoresContrast*100:0.2f}%")
            print(f"\tEM Score: {emScores*100:0.2f}%")
            print(f"\tEM Score (w/o original questions): {emScoresContrast*100:0.2f}%")
        if curPromptType and curPromptType not in promptPerfs.keys():
            promptPerfs[curPromptType] = {
                "failureEditTypes": [],
                "failureCases": [],
                "numEditTypes": {},
            }
        if trainSet not in promptPerfs[curPromptType].keys():
            promptPerfs[curPromptType][trainSet] = {
                "consistency": [],
                "accuracy": [],
                "accuracyContrast": [],
                "f1Score": [],
                "f1ScoreContrast": [],
                "emScore": [],
                "emScoreContrast": [],
                "numExamples": 0,
                "numContrastSets": 0,
                "numExamplesContrast": 0,
                "numMisses": {}
            }
        if curPromptType:
            promptPerfs[curPromptType][trainSet]["accuracy"].append(accuracy)
            promptPerfs[curPromptType][trainSet]["accuracyContrast"].append(accuracyContrast)
            promptPerfs[curPromptType][trainSet]["consistency"].append(consistency)
            promptPerfs[curPromptType][trainSet]["numExamples"] += totalCounts
            promptPerfs[curPromptType][trainSet]["numExamplesContrast"] += totalCountsContrast
            promptPerfs[curPromptType][trainSet]["numContrastSets"] += numQues
            promptPerfs[curPromptType][trainSet]["f1Score"].append(f1Scores)
            promptPerfs[curPromptType][trainSet]["f1ScoreContrast"].append(f1ScoresContrast)
            promptPerfs[curPromptType][trainSet]["emScore"].append(emScores)
            promptPerfs[curPromptType][trainSet]["emScoreContrast"].append(emScoresContrast)
            promptPerfs[curPromptType][trainSet]["numMisses"].update(numMisses)
            if dataset == "condaqa":
                promptPerfs[curPromptType]["failureEditTypes"].extend(failureEditTypes)
            promptPerfs[curPromptType]["failureCases"].extend(failureCases)
            if dataset == "condaqa":
                if "numEditTypes" not in promptPerfs[curPromptType].keys():
                    promptPerfs[curPromptType]["numEditTypes"] = numEditTypes
                else:
                    for et in numEditTypes.keys():
                        if et not in promptPerfs[curPromptType]["numEditTypes"].keys():
                            promptPerfs[curPromptType]["numEditTypes"][et] = 0
                        promptPerfs[curPromptType]["numEditTypes"][et] += numEditTypes[et]
        if not concise and not summaryOnly:
            if dataset == "condaqa":
                print(f"\tFailure by edit types:\n\t\tEditID: Number of Failures/Total no. of instances of editID (%)")
                vals, counts = np.unique(failureEditTypes, return_counts=True)
                for et in range(len(vals)):
                    editIDerr = counts[et]/numEditTypes[vals[et]]
                    print(f"\t\t{vals[et]}: {counts[et]}/{numEditTypes[vals[et]]} ({round(editIDerr*100,2)}%)")
            print(f"\tConfusion Cases:\n\t\t(Output, Label): Number of such cases")
            fcs = []
            for fc in failureCases:
                fcs.append(str((fc["output"], fc["label"])))
            vals, counts = np.unique(fcs, return_counts=True)
            CV = list(zip(counts, vals))
            CV.sort()
            CV = CV[-1::-1]
            for (c, v) in CV:
                info = (v[:50] + '...' + v[-50:]) if len(v) > 75 else v
                print(f"\t\t{info}: {c}")
        if not summaryOnly:
            print("++++++++++++++++++++++++++++++++")
    print("Performance Summary:")
    for  promptType in np.sort(list(promptPerfs.keys())):
        print(f"Prompt Type: {promptType}")
        consistency = 0
        accuracy = 0
        accuracyContrast = 0
        F1Score = 0
        F1ScoreContrast = 0
        EMScore = 0
        EMScoreContrast = 0
        statKeys = ["failureEditTypes", "failureCases", "numEditTypes"]
        for trainSet in promptPerfs[promptType].keys():
            if trainSet in  statKeys:
                continue
            curCons = promptPerfs[promptType][trainSet]["consistency"]
            consistency += np.sum(curCons)/len(curCons)
            curAcc = promptPerfs[promptType][trainSet]["accuracy"]
            accuracy += np.sum(curAcc)/len(curAcc)
            curAccContrast = promptPerfs[promptType][trainSet]["accuracyContrast"]
            accuracyContrast += np.sum(curAccContrast)/len(curAccContrast)
            curF1Score = promptPerfs[promptType][trainSet]["f1Score"]
            F1Score += np.sum(curF1Score)/len(curF1Score)
            curF1ScoreContrast = promptPerfs[promptType][trainSet]["f1ScoreContrast"]
            F1ScoreContrast += np.sum(curF1ScoreContrast)/len(curF1ScoreContrast)
            curEMScore = promptPerfs[promptType][trainSet]["emScore"]
            EMScore += np.sum(curEMScore)/len(curEMScore)
            curEMScoreContrast = promptPerfs[promptType][trainSet]["emScoreContrast"]
            EMScoreContrast += np.sum(curEMScoreContrast)/len(curEMScoreContrast)
            if not concise:
                numExamples = promptPerfs[promptType][trainSet]["numExamples"]
                numExamplesContrast = promptPerfs[promptType][trainSet]["numExamplesContrast"]
                numContrastSets = promptPerfs[promptType][trainSet]["numContrastSets"]
                print(f"\tTrain set {trainSet}:\n\t\tNo. of examples: {numExamples}\n\t\tNo. of examples (w/o original questions): {numExamplesContrast}\n\t\tNo. of contrast sets: {numContrastSets}\n\t\t******\n\t\tConsistency: {(np.sum(curCons)/len(curCons))*100:0.2f}%\n\t\tAccuracy: {(np.sum(curAcc)/len(curAcc))*100:0.2f}%\n\t\tAccuracy (w/o original questions): {(np.sum(curAccContrast)/len(curAccContrast))*100:0.2f}%\n\t\tF1 Score: {(np.sum(curF1Score)/len(curF1Score))*100:0.2f}%\n\t\tF1 Score (w/o original questions):{(np.sum(curF1ScoreContrast)/len(curF1ScoreContrast))*100:0.2f}%\n\t\tExact Match Score: {(np.sum(curEMScore)/len(curEMScore))*100:0.2f}%\n\t\tExact Match Score (w/o original questions):{(np.sum(curEMScoreContrast)/len(curEMScoreContrast))*100:0.2f}%")
                print(f"\t\tNo. of wrong  answers in contrast set : No. of such contrast sets")
                for k in promptPerfs[promptType][trainSet]["numMisses"].keys():
                    print("\t\t{:>37} : {}".format(k, promptPerfs[promptType][trainSet]["numMisses"][k]))
        numTrainSets = len(promptPerfs[promptType].keys()) - len(statKeys)
        consistency /= numTrainSets
        accuracy /= numTrainSets
        accuracyContrast /= numTrainSets
        F1Score /= numTrainSets
        F1ScoreContrast /= numTrainSets
        EMScore /= numTrainSets
        EMScoreContrast /= numTrainSets
        print(f"\tAverage Consistency: {round(consistency*100,2):0.2f}%")
        print(f"\tAverage Accuracy: {round(accuracy*100,2):0.2f}%")
        print(f"\tAverage Accuracy (w/o original questions): {round(accuracyContrast*100,2):0.2f}%")
        print(f"\tAverage F1 Score: {F1Score*100:0.2f}%")
        print(f"\tAverage F1 Score (w/o original questions): {F1ScoreContrast*100:0.2f}%")
        print(f"\tAverage Exact Match Score: {EMScore*100:0.2f}%")
        print(f"\tAverage Exact Match Score (w/o original questions): {EMScoreContrast*100:0.2f}%")
        if not concise:
            #Only for CondaQA
            if dataset == "condaqa":
                print(f"\tFailure by edit types:\n\t\tEditID: Number of Failures/Total no. of instances of editID (%)")
                vals, counts = np.unique(promptPerfs[promptType]["failureEditTypes"], return_counts=True)
                numEditTypes = promptPerfs[promptType]["numEditTypes"]
                for et in range(len(vals)):
                    editIDerr = counts[et]/numEditTypes[vals[et]]
                    print(f"\t\t{vals[et]}: {counts[et]}/{numEditTypes[vals[et]]} ({round(editIDerr*100,2)}%)")
            #Only for CondaQA
            print(f"\tConfusion Cases:\n\t\t(Output, Label): Number of such cases")
            fcs = []
            for fc in promptPerfs[promptType]["failureCases"]:
                fcs.append(str((fc["output"], fc["label"])))
            vals, counts = np.unique(fcs, return_counts=True)
            CV = list(zip(counts, vals))
            CV.sort()
            CV = CV[-1::-1]
            for (c, v) in CV:
                info = (v[:50] + '...' + v[-50:]) if len(v) > 75 else v
                print(f"\t\t{info}: {c}")
        print("++++++++++++++++++++++++++")
#---------------------------------------------------------------------------
if __name__ == '__main__':
    main()
#---------------------------------------------------------------------------