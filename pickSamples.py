import argparse 
import json
import numpy as np
import os
from pathlib import Path

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
    required=True
)

parser.add_argument(
    "-numSamples",
    type=int,
    help = "Number of samples in each set",
    required=True
)

parser.add_argument(
    "-chosenIndicesPath",
    type=str,
    help = "Path to npy file containing chosen indices",
)

parser.add_argument(
    "-dataset",
    choices = ["condaqa", "boolq", "drop", "ropes", "mctaco", "quoref", "imdb", "matres", "nlvr2"],
    required=True
)

parser.add_argument(
    "-extractQs",
    action="store_true",
    help="Set to extractQs mode to extract questions for (background, situation) [ROPES dataset only]"
)

parser.add_argument(
    "-bg",
    "--background",
    help="Background to extract questions for (background, situation) [ROPES dataset only]"
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
    help = "Path to json file containing original dataset (for quoref/IMDB dataset only)",
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
if args.chosenIndicesPath:
    chosenIndicesPath = args.chosenIndicesPath
if args.background:
    inputBG = args.background 
if args.situation:
    inputST = args.situation

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
#--------------------------------------------------------------------------------
def main():
    if dataset == "mctaco" or dataset == "ropes" or dataset == "quoref" or dataset == "imdb":

        if dataset == "mctaco" and extractQs:
            if inputBG == None:
                raise ValueError("Background must be provided to extract questions!")
            passages = {}
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
            

        with open(inputFile,"r", encoding='utf-8-sig') as f:
            data = json.load(f)
        passages = {}

        if dataset == "ropes":    
            data = data["data"][0]["paragraphs"]
            for i in range(len(data)):
                background = data[i]["background"]
                situation = data[i]["situation"]
                bgst = (background + situation)
                if bgst  not in passages.keys():
                    passages[bgst] = {
                        "background": background,
                        "situation": situation,
                        "qas": []
                    }
                passages[bgst]["qas"].extend(data[i]["qas"])
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

        passKeys = np.array(list(passages.keys()))
        if chosenIndicesPath == None:
            chosenIndices = np.random.choice(len(passKeys), size=numSamples*numSets, replace=False)
            np.save(outputPath+"_"+inputFile.split("/")[-1].split(".")[0]+"_chosenIndices.npy",chosenIndices)
        else: 
            chosenIndices = np.load(chosenIndicesPath)
        chosenData = passKeys[chosenIndices]
        for s in range(numSets):
            chosenPasses = []
            for i in range(numSamples):
                chosenPasses.append(passages[chosenData[s*numSamples+i]])
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






