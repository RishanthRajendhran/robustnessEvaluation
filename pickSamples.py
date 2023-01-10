#For MCTACO dataset, after samples from the perturbed dataset are chosen, 
# we'd have to manually find matching passages in the original dataset
# One could use the extractQ mode to retrieve matching passages giving 
# appropriate passage cues

import argparse 
import json
import numpy as np
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser()

parser.add_argument(
    "-input",
    help = "Path to json file containing dataset",
    required=True
)

parser.add_argument(
    "-output",
    help = "directory path to save output file; in case of extractQs mode this path should contain file name",
    required=True
)

parser.add_argument(
    "-numSets",
    type=int,
    help = "Number of sets",
    default=None
)

parser.add_argument(
    "-numSamples",
    type=int,
    help = "Number of samples in each set",
    default=None
)

parser.add_argument(
    "-chosenIndicesPath",
    type=str,
    help = "Path to npy file containing chosen indices",
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
    "-extractQs",
    action="store_true",
    help="Set to extractQs mode to extract questions for (background, situation) [ROPES/MCTACO dataset only]"
)

parser.add_argument(
    "-bg",
    "--background",
    help="Background to extract questions for (background, situation) [ROPES/MCTACO dataset only]"
)

parser.add_argument(
    "-st",
    "--situation",
    help="Situation to extract questions for (background, situation) [ROPES dataset only]",
    default=""
)

#For quoref
parser.add_argument(
    "-inputOriginal",
    help = "Path to json file containing original dataset (for ropes/quoref/IMDB dataset only)",
)

parser.add_argument(
    "-train",
    action="store_true",
    help="Boolean flag to indicate that code should generate train data",
)

args = parser.parse_args()

inputFile = args.input 
outputPath = args.output
numSets = args.numSets
numSamples = args.numSamples
chosenIndicesPath = None 
dataset = args.dataset
extractQs = args.extractQs
inputOriginal = args.inputOriginal
train = args.train
if args.chosenIndicesPath:
    chosenIndicesPath = args.chosenIndicesPath
if args.background:
    inputBG = args.background 
inputST = args.situation

if not extractQs:
    if not numSets:
        raise RuntimeError("Missing command line arguments!")

file_exists = os.path.exists(inputFile)
if not file_exists:
    print(f"{inputFile} is an invalid input file path!")
    exit(0)
if extractQs:
    path = Path(outputPath)
    if not path.is_file():
        print(f"{outputPath} is not a file!")
        exit(0)
else: 
    if not os.path.exists(outputPath):
        print(f"{outputPath} is an invalid directory path!")
        exit(0)
#--------------------------------------------------------------------------------
# For ropes
def findCommonStartStr(s1, s2):
    if len(s1) == 0 or len(s2) == 0:
        return ""
    mid = min(len(s1), len(s2))//2
    if s1[:mid+1] == s2[:mid+1]:
        return s1[:mid+1] + findCommonStartStr(s1[mid+1:], s2[mid+1:])
    else: 
        if mid == 0:
            return ""
        return findCommonStartStr(s1[:mid], s2[:mid])
#--------------------------------------------------------------------------------
# For ropes
def retrieveQuestions(fileName, passages, bg, st=""):
    with open(fileName, 'w') as sys.stdout:
        out = []
        for k in passages.keys():
            if bg in k:
                out.append({
                    "background": k,
                    "situations": passages[k]
                })
        for i in range(len(out)):
            background = out[i]["background"]
            situations = out[i]["situations"]
            for j in situations.keys():
                if len(st) and st not in j:
                    continue
                print(f"PassageID: {i}")
                print(f"Passage: Background: {background} Situation: {j}", end="")
                print("\n")
                print("Questions:")
                qas = situations[j]
                for k in range(len(qas)):
                    que = qas[k]["question"]
                    answers = qas[k]["answers"]
                    print(f"QuestionID: {k}")
                    print(f"Question: {que}")
                    ans = []
                    for l in range(len(answers)):
                        for key in answers[l].keys():
                            ans.append(answers[l][key])
                    finalAns = ", ".join(ans)
                    print(f"Answer: {finalAns}")
                    print("Explanation: ",end="")
                    print("\n")
        print("--------------------------------------------------")
#--------------------------------------------------------------------------------
#For MCTACO dataset
def getQuestions(fileName, data, ps, pid="0_0"):
    with open(fileName, 'w') as sys.stdout:
        out = []
        firstMatch = None
        for d in data: 
            if "sentence1" not in d.keys():
                continue
            if ps in d["sentence1"]:
                if firstMatch == None: 
                    firstMatch = int(d["index"])
                out.append({
                "PassageID": pid,
                "sentence1": d["sentence1"],
                "QuestionID": str(int(d["index"])-firstMatch),
                "sentence2": "Is this the answer: " + d["sentence2"] + "?",
                "label": d["label"],
                })
        print("Questions:")
        for o in out:
            qID = o["QuestionID"]
            que = o["sentence1"] + " " + o["sentence2"]
            answer = o["label"]
            print(f"QuestionID: {qID}")
            print(f"Question: {que}")
            print(f"Answer: {answer}")
            print("Explanation: ",end="")
            print("\n")
        print("--------------------------------------------------")
#--------------------------------------------------------------------------------
def main():
    if dataset == "mctaco" or dataset == "ropes" or dataset == "quoref" or dataset == "imdb" or dataset == "perspectrum":

        with open(inputFile,"r", encoding='utf-8-sig') as f:
            data = json.load(f)
        passages = {}

        if dataset == "ropes" and extractQs:
            if inputBG == None:
                raise ValueError("Background must be provided to extract questions!")
            data = data["data"][0]["paragraphs"]
            for i in range(len(data)):
                background = data[i]["background"]
                situation = data[i]["situation"]
                if background not in passages.keys():
                    passages[background] = {}
                if situation not in passages[background].keys():
                    passages[background][situation] = []
                passages[background][situation].extend(data[i]["qas"])
            retrieveQuestions(outputPath, passages, inputBG, inputST)
            exit(0)
        
        if dataset == "mctaco" and extractQs:
            if inputBG == None:
                raise ValueError("Passage clue must be provided to extract questions!")
            getQuestions(outputPath, data, inputBG)
            exit(0)

        if dataset == "ropes":    
            data = data["data"][0]["paragraphs"]
            if not train:
                with open(inputOriginal,"r", encoding='utf-8-sig') as f:
                    dataOriginal = json.load(f)
                dataOriginal = dataOriginal["data"][0]["paragraphs"]
            else: 
                dataOriginal = data
            oriPassages = {}
            for i in range(len(dataOriginal)):
                background = dataOriginal[i]["background"]
                situation = dataOriginal[i]["situation"]
                if background not in oriPassages.keys():
                    oriPassages[background] = {
                        "background": background,
                        "situation": situation,
                        "qas": []
                    }
                oriPassages[background]["qas"].extend(dataOriginal[i]["qas"])
            if train: 
                passages = oriPassages
                for p in passages.keys():
                    passages[p]["qas"] = np.random.choice(passages[p]["qas"], 1).tolist()
            else:
                perPassages = {}
                for i in range(len(data)):
                    background = data[i]["background"]
                    situation = data[i]["situation"]
                    if background not in perPassages.keys():
                        perPassages[background] = {
                            "situation": situation,
                            "qas": []
                        }
                    perPassages[background]["qas"].extend(data[i]["qas"])
                for bg in oriPassages.keys():
                    if bg in perPassages.keys():
                        commonSituation = findCommonStartStr(oriPassages[bg]["situation"], perPassages[bg]["situation"])
                        fullStopPos = len(commonSituation)-1
                        while fullStopPos >=0:
                            if commonSituation[fullStopPos] == ".":
                                commonSituation = commonSituation[:fullStopPos+1]
                                break 
                            fullStopPos -= 1
                        if fullStopPos < 0:
                            commonSituation = ""
                        if len(commonSituation) == len(oriPassages[bg]["situation"]) or len(commonSituation) == len(perPassages[bg]["situation"]):
                            continue
                        if (bg + commonSituation) not in passages.keys():
                            passages[(bg + commonSituation)] = {
                                "background": bg,
                                "situation": commonSituation,
                                "qas": []
                            }
                        for q in oriPassages[bg]["qas"]:
                            newQ = q.copy()
                            newQ["question"] = oriPassages[bg]["situation"][len(commonSituation):].strip() + " " + newQ["question"].strip()
                            passages[(bg + commonSituation)]["qas"].append(newQ)
                        for q in perPassages[bg]["qas"]:
                            newQ = q.copy()
                            newQ["question"] = perPassages[bg]["situation"][len(commonSituation):].strip() + " " + newQ["question"].strip()
                            passages[(bg + commonSituation)]["qas"].append(newQ)  
        elif dataset == "mctaco":
            for i in range(len(data)):
                if "sentence1" not in data[i].keys():
                    continue
                if data[i]["sentence1"] not in passages.keys():
                    passages[data[i]["sentence1"]] = {
                        "sentence1": data[i]["sentence1"],
                        "qas": []
                    }
                passages[data[i]["sentence1"]]["qas"].append({
                    "sentence2": data[i]["sentence2"],
                    "label": data[i]["label"]
                })
        elif dataset == "quoref":
            data = data["data"]
            with open(inputOriginal,"r", encoding='utf-8-sig') as f:
                dataOriginal = json.load(f)
            dataOriginal = dataOriginal["data"]
            passages = {}
            for i in range(len(dataOriginal)):
                for p in dataOriginal[i]["paragraphs"]:
                    if p["context_id"] not in passages.keys():
                        passages[p["context_id"]] = {
                            "context": p["context"],
                            "qas": []
                        }
                    passages[p["context_id"]]["qas"].extend(p["qas"])
            for i in range(len(data)):
                for p in data[i]["paragraphs"]:
                    if p["context_id"] in passages.keys():
                        passages[p["context_id"]]["qas"].extend(p["qas"])
        elif dataset == "imdb":
            with open(inputOriginal,"r", encoding='utf-8-sig') as f:
                dataOriginal = json.load(f)
            passages = {}
            counter = 0
            dataConsolidated = list(zip(data, dataOriginal))
            for i in range(len(dataConsolidated)):
                reviewPer = dataConsolidated[i][0]["Text"]
                reviewOri = dataConsolidated[i][1]["Text"]
                sentimentPer = dataConsolidated[i][0]["Sentiment"]
                sentimentOri = dataConsolidated[i][1]["Sentiment"]
                sentsPer = [s.strip() for s in reviewPer.split(".") if len(s.strip())]
                sentsOri = [s.strip() for s in reviewOri.split(".") if len(s.strip())]
                psg = ""
                numSentPsg = 0
                for i in range(min(len(sentsPer), len(sentsOri))):
                    if sentsPer[i] == sentsOri[i]:
                        psg += sentsPer[i] + ". "
                        numSentPsg += 1
                    else: 
                        break 
                psg = psg.strip()
                if len(psg) or psg in passages.keys():
                    passages[psg] = {
                        "passage": psg,
                        "qas": []
                    }
                    passages[psg]["qas"].append({
                        "question": (". ".join(sentsPer[numSentPsg:])).strip(),
                        "answer": sentimentPer
                    })
                    passages[psg]["qas"].append({
                        "question": (". ".join(sentsOri[numSentPsg:])).strip(),
                        "answer": sentimentOri
                    })
                else:
                    while str(counter) in passages.keys():
                        counter += 1
                    passages[str(counter)] = {
                        "passage": "",
                        "qas": []
                    }
                    passages[str(counter)]["qas"].append({
                        "question": (". ".join(sentsPer)).strip(),
                        "answer": sentimentPer
                    })
                    passages[str(counter)]["qas"].append({
                        "question": (". ".join(sentsOri)).strip(),
                        "answer": sentimentOri
                    })
                    counter += 1
        elif dataset == "perspectrum":
            passages = {}
            for i in range(len(data)):
                claims = []
                claims.append(data[i]["original_claim"])
                claims.append(data[i]["contrast_claim"])
                claims.sort()
                claims = tuple(claims)
                if claims not in passages.keys():
                    passages[claims] = {
                        "original_claim": data[i]["original_claim"],
                        "contrast_claim": data[i]["contrast_claim"],
                        "perspectives": []
                    }
                passages[claims]["perspectives"].append({
                    "perspective": data[i]["perspective"],
                    "original_stance_label": data[i]["original_stance_label"],
                    "contrast_stance_label": data[i]["contrast_stance_label"]
                })

        passKeys = np.array(list(passages.keys()))
        global numSamples
        if numSamples == None:
            numSamples = len(passKeys)//numSets
        if chosenIndicesPath == None:
            chosenIndices = np.random.choice(len(passKeys), size=numSamples*numSets, replace=False)
            np.save(outputPath+"_"+inputFile.split("/")[-1].split(".")[0]+"_chosenIndices.npy",chosenIndices)
        else: 
            chosenIndices = np.load(chosenIndicesPath)
        chosenData = passKeys[chosenIndices]
        for s in range(numSets):
            chosenPasses = []
            for i in range(numSamples):
                if dataset == "ropes":
                    chosenPasses.append(passages[chosenData[s*numSamples+i]])
                else: 
                    chosenPasses.append(passages[tuple(chosenData[s*numSamples+i])])
            jsonObj = json.dumps(chosenPasses, indent=4)
            with open(outputPath+"_"+inputFile.split("/")[-1].split(".")[0]+"_"+str(s)+".json","w") as o:
                o.write(jsonObj)
    else:
        if dataset == "condaqa" or dataset == "nlvr2":
            data = []
            for line in open(inputFile,"r"):
                data.append(json.loads(line))
        elif dataset == "boolq" or dataset == "drop" or dataset == "matres":
            with open(inputFile, "r") as f:
                data = json.load(f)
            if dataset == "boolq":
                data = data["data"][1:]
        data = np.array(data)
        if numSamples == None:
            numSamples = len(data)//numSets

        if chosenIndicesPath == None:
            chosenIndices = np.random.choice(len(data), size=numSamples*numSets, replace=False)
            np.save(outputPath+"_"+inputFile.split("/")[-1].split(".")[0]+"_chosenIndices.npy",chosenIndices)
        else: 
            chosenIndices = np.load(chosenIndicesPath)
        chosenData = data[chosenIndices]
        for s in range(numSets):
            jsonObj = json.dumps(chosenData[s*numSamples:s*numSamples+numSamples].tolist(), indent=4)
            with open(outputPath+"_"+inputFile.split("/")[-1].split(".")[0]+"_"+str(s)+".json","w") as o:
                o.write(jsonObj)
#--------------------------------------------------------------------------------
if __name__ == "__main__":
    main()

