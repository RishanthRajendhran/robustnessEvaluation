import json
import argparse
import numpy as np
import os
import glob

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
        "perspectrum"
    ],
    required=True,
)

args = parser.parse_args()
outputFiles = args.outputFiles
isDirectory = args.isDirectory
concise = args.concise
summaryOnly = args.summaryOnly
dataset = args.dataset

if isDirectory:
    jsonDirName = outputFiles[0]
    jsonPattern = os.path.join(jsonDirName, '*.json')
    outputFiles = glob.glob(jsonPattern)
#---------------------------------------------------------------------------
def extractAnswer(answer, isOut):
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
    return answer
#---------------------------------------------------------------------------
def main():
    promptPerfs = {}
    for outputFile in outputFiles:
        passages = {}
        data = []   
        numEditTypes = {}
        for  line in open(outputFile, "r"):
            data.extend(json.loads(line))
        for d in data:
            answer = d["output"]
            answer = extractAnswer(answer, True)
            corrAnswer = d["label"]
            corrAnswer = extractAnswer(corrAnswer, False)
            passId = d["PassageID"]
            questionID = d["QuestionID"]
            if dataset == "condaqa":
                if d["PassageEditID"] not in numEditTypes.keys():
                    numEditTypes[d["PassageEditID"]] = 1
                else:
                    numEditTypes[d["PassageEditID"]] += 1
            if passId not in passages.keys():
                if dataset in ["condaqa", "perspectrum", "mctaco", "ropes"]:
                    passages[passId] = {}
                else:
                    passages[passId] = {
                        "score": 0,
                        "count": 0,
                        "failureEditTypes": [],
                        "failureCases": [],
                    }
            if dataset in ["condaqa", "perspectrum", "mctaco", "ropes"]:
                if questionID not in passages[passId].keys():
                    passages[passId][questionID] = {
                        "score": 0,
                        "count": 0,
                        "failureEditTypes": [],
                        "failureCases": [],
                    }
            if dataset in ["condaqa", "perspectrum", "mctaco", "ropes"]:
                passages[passId][questionID]["count"] += 1
            else:
                passages[passId]["count"] += 1
            if answer == corrAnswer or ((" " + corrAnswer + " ") in answer and abs(len(corrAnswer)-len(answer))<=6) or ((" " + answer + " ") in  corrAnswer and abs(len(corrAnswer)-len(answer))<=6):
                if dataset in ["condaqa", "perspectrum", "mctaco", "ropes"]:
                    passages[passId][questionID]["score"] += 1
                else: 
                    passages[passId]["score"] += 1
            else:
                #Are answer and corrAnswer permutations of each other?
                sortedAns = np.sort([x for x in answer.split(" ") if len(x) and x.isalnum()])
                sortedCorrAns = np.sort([x for x in corrAnswer.split(" ") if len(x) and x.isalnum()])
                if len(sortedAns) == len(sortedCorrAns) and (sortedAns == sortedCorrAns).all():
                    if dataset in ["condaqa", "perspectrum", "mctaco", "ropes"]:
                        passages[passId][questionID]["score"] += 1
                    else: 
                        passages[passId]["score"] += 1
                else:
                    if dataset in ["condaqa", "perspectrum", "mctaco", "ropes"]:
                        if dataset == "condaqa":
                            passages[passId][questionID]["failureEditTypes"].append(d["PassageEditID"])
                        passages[passId][questionID]["failureCases"].append({"output":answer, "label":corrAnswer})
                    else:
                        passages[passId]["failureCases"].append({"output":answer, "label":corrAnswer})
        accuracy = 0
        consistency = 0
        totalCounts = 0
        numQues = 0
        failureEditTypes = []
        failureCases = []
        for pID in passages.keys():
            numQues += len(passages[pID].keys())
            if dataset in ["condaqa", "perspectrum", "mctaco", "ropes"]:
                for qID in passages[pID].keys():
                    accuracy += passages[pID][qID]["score"]
                    totalCounts += passages[pID][qID]["count"]
                    if passages[pID][qID]["score"] == passages[pID][qID]["count"]:
                        consistency += 1
                    if dataset == "condaqa":
                        failureEditTypes.extend(passages[pID][qID]["failureEditTypes"])
                    failureCases.extend(passages[pID][qID]["failureCases"])
            else:
                accuracy += passages[pID]["score"]
                totalCounts += passages[pID]["count"]
                if passages[pID]["score"] == passages[pID]["count"]:
                    consistency += 1
                failureCases.extend(passages[pID]["failureCases"])
        accuracy /= totalCounts
        consistency /= numQues

        if not summaryOnly:
            print(f"{outputFile}")
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
            print(f"\tAccuracy: {round(accuracy*100,2)}%")
            print(f"\tConsistency: {round(consistency*100,2)}%")
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
            }
        if curPromptType:
            promptPerfs[curPromptType][trainSet]["accuracy"].append(accuracy)
            promptPerfs[curPromptType][trainSet]["consistency"].append(consistency)
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
            print(f"\tFailure Cases:\n\t\t(Output, Label): Number of Failures")
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
        statKeys = ["failureEditTypes", "failureCases", "numEditTypes"]
        for trainSet in promptPerfs[promptType].keys():
            if trainSet in  statKeys:
                continue
            curCons = promptPerfs[promptType][trainSet]["consistency"]
            consistency += np.sum(curCons)/len(curCons)
            curAcc = promptPerfs[promptType][trainSet]["accuracy"]
            accuracy += np.sum(curAcc)/len(curAcc)
            if not concise:
                print(f"\tTrain set {trainSet}:\n\t\tConsistency: {np.sum(curCons)/len(curCons)}\n\t\tAccuracy: {np.sum(curAcc)/len(curAcc)}")
        numTrainSets = len(promptPerfs[promptType].keys()) - len(statKeys)
        consistency /= numTrainSets
        accuracy /= numTrainSets
        print(f"\tAverage Consistency: {round(consistency*100,2)}%")
        print(f"\tAverage Accuracy: {round(accuracy*100,2)}%")
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
            print(f"\tFailure Cases:\n\t\t(Output, Label): Number of Failures")
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
 



