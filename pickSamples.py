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
import glob
import regex as re
from html.parser import HTMLParser
from conllu import parse

#For MATRES dataset in train mode
class MyHTMLParser(HTMLParser):
    def __init__(self):
        super(MyHTMLParser, self).__init__()
        self.document = {"instances":[], "events": [], "text": ""}
        self.curOpenTag = None
        self.textTagOpen = False
    def handle_starttag(self, tag, attrs):
        self.curOpenTag = tag
        if tag == "text":
            self.textTagOpen = True
        elif tag == "event":
            EID = None 
            for a in attrs:
                if a[0] == "eid":
                    EID = a[1]
            if not EID:
                print(f"Event is missing EID! This should not haave happened!")
                exit(0)
            self.document["events"].append({
                "eid": EID,
                "loc": len(self.document["text"])+1
            })
        elif tag == "makeinstance":
            EID, EIID = None, None
            for a in attrs:
                if a[0] == "eventid":
                    EID = a[1]
                elif a[0] == "eiid":
                    EIID = a[1]
            if not EID or not EIID:
                print(f"Makeinstance is missing EID/EIID in {attrs}! This should not have happened!")
                exit(0)
            self.document["instances"].append({
                "eid": EID,
                "eiid": EIID,
            })

    def handle_endtag(self, tag):
        self.curOpenTag = None
        if tag == "text":
            self.textTagOpen = False

    def handle_data(self, data):
        if self.curOpenTag == "docid":
            self.document["docid"] = data 
        elif self.curOpenTag == "text":
            self.document["text"] = data
            self.document["text"] = self.document["text"].strip().replace("\n", "")
        elif self.textTagOpen:
            self.document["text"] = self.document["text"].strip().replace("\n", "") + " " + data
            self.document["text"] = self.document["text"].strip().replace("\n", "")
    
    def getDocDict(self):
        return self.document
#For MATRES dataset in train mode

parser = argparse.ArgumentParser()

parser.add_argument(
    "-input",
    help = "Path to json (or txt file in case of MATRES dataset in train mode or conllu file in case of UDPARSING dataset) file containing dataset",
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
                "perspectrum",
                "udparsing",
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
    help = "Path to json file (or directory in case of MATRES dataset in train mode or conllu file in case of UDPARSING dataset) containing original dataset (for DROP/ropes/quoref/IMDB/mctaco/MATRES/UD Parsing dataset only)",
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
        raise RuntimeError("Missing one of the following two command line arguments: numSets or extractQs!")

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
    if dataset in ["mctaco", "ropes", "quoref", "imdb", "perspectrum", "udparsing"]:

        if dataset != "udparsing":
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
                if dataOriginal[i]["background"] not in oriPassages.keys():
                    oriPassages[dataOriginal[i]["background"]] = {
                        "background": dataOriginal[i]["background"],
                        "qas": []
                    }
                for q in dataOriginal[i]["qas"]:
                    oriPassages[dataOriginal[i]["background"]]["qas"].append({
                        "background": dataOriginal[i]["background"],
                        "situation": dataOriginal[i]["situation"],
                        "question": q["question"],
                        "id": q["id"],
                        "answers": q["answers"],
                    })
            if train: 
                passages = oriPassages
                for p in passages.keys():
                    passages[p] = np.random.choice(passages[p], 1).tolist()
            else:
                passages = {}
                for i in range(len(dataOriginal)):
                    for q in dataOriginal[i]["qas"]:
                        if q["id"] not in passages.keys():
                            passages[q["id"]] = {
                                "background": dataOriginal[i]["background"],
                                "situation": dataOriginal[i]["situation"],
                                "qas": []
                            }
                        passages[q["id"]]["qas"].append({
                            "background": dataOriginal[i]["background"],
                            "situation": dataOriginal[i]["situation"],
                            "question": q["question"],
                            "id": q["id"],
                            "answers": q["answers"],
                            "isOriginal": True,
                        })
                for i in range(len(data)):
                    for q in data[i]["qas"]:
                        if q["id"] in passages.keys():
                            passages[q["id"]]["qas"].append({
                                "background": data[i]["background"],
                                "situation": data[i]["situation"],
                                "question": q["question"],
                                "id": q["id"],
                                "answers": q["answers"],
                                "isOriginal": False,
                            })
                            # if data[i]["background"].lower() != passages[q["id"]]["background"].strip().replace("\n","").lower():
                            #     print("Naan kettadhu:")
                            #     print("\t{}".format(data[i]["background"]))
                            #     print("Avar enakku kuduthadhu")
                            #     print("\t{}".format(passages[q["id"]]["background"]))
                            #     c += 1
                            # else:
                            #     c2 += 1 
                #             # if q["id"]+"_perturbed" not in passages.keys():
                #             #     passages[q["id"]+"_perturbed"] = {
                #             #         "background": data[i]["background"],
                #             #         "situation": data[i]["situation"],
                #             #         "qas": []
                #             #     }
                #             # passages[q["id"]+"_perturbed"]["qas"].append({
                #             #     "background": data[i]["background"],
                #             #     "situation": data[i]["situation"],
                #             #     "question": q["question"],
                #             #     "id": q["id"],
                #             #     "answers": q["answers"],
                #             #     "isOriginal": False,
                #             # })
        elif dataset == "mctaco":
            if not train:
                with open(inputOriginal,"r", encoding='utf-8-sig') as f:
                    dataOriginal = json.load(f)
                #Extract questions from original dataset
                for i in range(len(dataOriginal)):
                    if "sentence1" not in dataOriginal[i].keys():
                        continue
                    if "sentence2" not in dataOriginal[i].keys():
                        continue
                    if dataOriginal[i]["index"] not in passages.keys():
                        passages[dataOriginal[i]["index"]] = {
                            "sentence1": dataOriginal[i]["sentence1"],
                            "qas": []
                        }
                    passages[dataOriginal[i]["index"]]["qas"].append({
                        "sentence2": dataOriginal[i]["sentence2"],
                        "label": dataOriginal[i]["label"]
                    })
                #Extract questions from perturbed dataset with
                #matching indices
                for i in range(len(data)):
                    if data[i]["index"] not in passages.keys():
                        continue
                    passages[data[i]["index"]]["qas"].append({
                        "sentence2": data[i]["sentence2"],
                        "label": data[i]["label"]
                    })
            else:
                newPassages = {}
                for i in range(len(data)):
                    if "sentence1" not in data[i].keys() or "sentence2" not in data[i].keys():
                        continue
                    if data[i]["sentence1"] not in newPassages.keys():
                        newPassages[data[i]["sentence1"]] = {}
                    if data[i]["sentence2"] not in newPassages[data[i]["sentence1"]].keys():
                        newPassages[data[i]["sentence1"]][data[i]["sentence2"]] = []
                    newPassages[data[i]["sentence1"]][data[i]["sentence2"]].append((data[i]["answer"], data[i]["label"]))
                i = 0
                for pkey in newPassages.keys():
                    passQList = list(newPassages[pkey].keys())
                    qkey = np.random.randint(len(newPassages[pkey]))
                    qChoice = passQList[qkey]
                    akey = np.random.randint(len(newPassages[pkey][qChoice]))
                    aChoice = newPassages[pkey][passQList[qkey]][akey]
                    passages[str(i)] = {
                        "sentence1": pkey + " " + qChoice,
                        "qas": [ 
                            {
                                "sentence2": aChoice[0],
                                "label": aChoice[1]
                            }
                        ]
                    }
                    i += 1
        elif dataset == "quoref":
            data = data["data"]
            if not train:
                with open(inputOriginal,"r", encoding='utf-8-sig') as f:
                    dataOriginal = json.load(f)
                dataOriginal = dataOriginal["data"]
            else: 
                dataOriginal = data
            if train:
                passages = {}
                for i in range(len(dataOriginal)):
                    for p in dataOriginal[i]["paragraphs"]:
                        if p["context_id"] not in passages.keys():
                            passages[p["context_id"]] = {
                                "context": p["context"],
                                "qas": []
                            }
                        passages[p["context_id"]]["qas"].extend(p["qas"])
                for p in passages.keys():
                    passages[p]["qas"] = np.random.choice(passages[p]["qas"],1).tolist()
            else:
                oriQues = {}
                for i in range(len(dataOriginal)):
                    for p in dataOriginal[i]["paragraphs"]:
                        for q in p["qas"]:
                            if q["id"] not in oriQues.keys():
                                oriQues[q["id"]] = q 
                            else:
                                print("qID: {} already seen! This should not have happened!")
                passages = {}
                for i in range(len(data)):
                    for p in data[i]["paragraphs"]:
                        for q in p["qas"]:
                            if q["original_id"] not in passages.keys():
                                passages[q["original_id"]] = {
                                    "context_id": p["context_id"],
                                    "context": p["context"],
                                    "qas": []
                                }
                            passages[q["original_id"]]["qas"].append(q)
                for oriID in passages.keys():
                    if oriID in oriQues.keys():
                        passages[oriID]["qas"].append(oriQues[oriID])
                    else: 
                        print("Could not find original question with id: {}!".format(oriID))
        elif dataset == "imdb":
            if train:
                passages = {}
                for i in range(len(data)):
                    passages[i] = {
                        "passage": data[i]["Text"],
                        "qas": [
                            {
                                "question": "",
                                "answer": data[i]["Sentiment"]
                            }
                        ]
                    }
            else:
                with open(inputOriginal,"r", encoding='utf-8-sig') as f:
                    dataOriginal = json.load(f)
                passages = {}
                counter = 0
                if len(data) != len(dataOriginal):
                    print("Size of input file and the original input file do not match!")
                    exit(0)
                for i in range(len(dataOriginal)):
                    if "Text" not in dataOriginal[i].keys() or "Text" not in data[i].keys():
                        continue
                    if "Sentiment" not in dataOriginal[i].keys() or "Sentiment" not in data[i].keys():
                        continue

                    passages[str(i)] = [{
                        "passage": dataOriginal[i]["Text"],
                        "qas": [{
                            "question": "What is the sentiment of this review: Positive or Negative?",
                            "answer": dataOriginal[i]["Sentiment"]
                        }]
                    }, {
                        "passage": data[i]["Text"],
                        "qas": [{
                            "question": "What is the sentiment of this review: Positive or Negative?",
                            "answer": data[i]["Sentiment"]
                        }]
                    }]
        elif dataset == "perspectrum":
            if train: 
                passages = {}
                for i in range(len(data)):
                    if data[i]["cId"] not in passages.keys():
                        passages[data[i]["cId"]] = {
                            "cId": data[i]["cId"],
                            "claim": data[i]["claim"],
                            "perspectives": []
                        }
                    passages[data[i]["cId"]]["perspectives"].append(data[i])
                for k in passages.keys():
                    passages[k]["perspectives"] = [passages[k]["perspectives"][np.random.choice(len(passages[k]["perspectives"]))]]
            else:
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
        elif dataset == "udparsing":
            if train:
                with open(inputFile, "r", encoding="utf-8") as f:
                    data = f.read()
                passages = {}
                for tree in parse(data):
                    ADPfound = False 
                    for i in range(len(tree)):
                        if tree[i]["upos"] == "ADP":
                            ADPfound = True
                            break 
                    if not ADPfound:
                        continue
                    sentID = tree.metadata["sent_id"]
                    if sentID not in passages.keys():
                        passages[sentID] = []
                    passages[sentID].append({
                        "sent_id": sentID,
                        "text": tree.metadata["text"],
                        "tokens": list(tree)
                    })
                for sentID in passages.keys():
                    passages[sentID] = passages[sentID][np.random.choice(len(passages[sentID]))]
            else:
                with open(inputFile, "r", encoding="utf-8") as f:
                    dataAltered = f.read()
                with open(inputOriginal, "r", encoding="utf-8") as f:
                    dataOriginal = f.read()
                passages = {}

                treesOri, treesAlt = parse(dataOriginal), parse(dataAltered)
                for (i, (treeOri, treeAlt)) in enumerate(zip(treesOri, treesAlt)):
                    sentID = treeOri.metadata["sent_id"]
                    if treeAlt.metadata["sent_id"] != (sentID+"***"):
                        print("sent_id's dont match! This should not have happened!")
                        exit(0)
                    passages[sentID+"_"+str(i)] = []
                    tokensOri = list(treeOri)
                    tokensAlt = list(treeAlt)
                    oriPos, altPos = None, None
                    for j in range(len(tokensOri)):
                        if tokensOri[j]["form"] != tokensAlt[j]["form"]:
                            oriPos = j
                            if tokensOri[oriPos]["upos"] != "ADP":
                                while oriPos>=0:
                                    if tokensOri[oriPos]["upos"] == "ADP":
                                        break 
                                    oriPos -= 1
                                if oriPos < 0:
                                    oriPos = j+1
                                    while oriPos<len(tokensOri):
                                        if tokensOri[oriPos]["upos"] == "ADP":
                                            break 
                                        oriPos += 1
                                    if oriPos>=len(tokensOri):
                                        oriPos = None
                            altPos = j
                            if tokensAlt[altPos]["upos"] != "ADP":
                                while altPos>=0:
                                    if tokensAlt[altPos]["upos"] == "ADP":
                                        break 
                                    altPos -= 1
                                if altPos < 0:
                                    altPos = j+1
                                    while altPos<len(tokensAlt):
                                        if tokensAlt[altPos]["upos"] == "ADP":
                                            break 
                                        altPos += 1
                                    if altPos>=len(tokensAlt):
                                        altPos = None
                                break
                    if oriPos == None: 
                        raise RuntimeError("Could not find perturbed ADP in original sentence!")
                    if altPos == None: 
                        raise RuntimeError("Could not find perturbed ADP in altered sentence!")
                    passages[sentID+"_"+str(i)].append({
                        "sent_id": sentID,
                        "text": treeOri.metadata["text"],
                        "tokens": list(treeOri),
                        "ADPid": tokensOri[oriPos]["id"],
                        "parentADPid": tokensOri[oriPos]["id"]
                    })
                    passages[sentID+"_"+str(i)].append({
                        "sent_id": sentID,
                        "text": treeAlt.metadata["text"],
                        "tokens": list(treeAlt),
                        "ADPid": tokensAlt[altPos]["id"],
                        "parentADPid": tokensOri[oriPos]["id"]
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
                if dataset in ["ropes", "quoref", "mctaco", "imdb", "udparsing"] or (train and dataset == "perspectrum"):
                    chosenPasses.append(passages[chosenData[s*numSamples+i]])
                else: 
                    chosenPasses.append(passages[tuple(chosenData[s*numSamples+i])])
            jsonObj = json.dumps(chosenPasses, indent=4)
            with open(outputPath+"_"+inputFile.split("/")[-1].split(".")[0]+"_"+str(s)+".json","w") as o:
                o.write(jsonObj)
    else:
        if dataset == "condaqa" or (dataset == "boolq" and train):
            data = []
            for line in open(inputFile,"r"):
                data.append(json.loads(line))
        elif dataset == "boolq" or dataset == "drop" or (dataset == "matres" and not train):
            with open(inputFile, "r") as f:
                data = json.load(f)
            if dataset == "boolq":
                data = data["data"][1:]
        if dataset == "matres":
            if train:
                if not inputFile.endswith(".txt"):
                    print(f"{inputFile} is not a valid txt file!")
                    exit(0)
                with open(inputFile, "r") as f:
                    data = [s.split("\t") for s in list(f.readlines())]
                docIDs = []
                for d in data:
                    if d[0] not in docIDs:
                        docIDs.append(d[0])
                if not os.path.isdir(inputOriginal):
                    print(f"{inputOriginal} is not a valid directory path!")
                    exit(0)
                tmlDirName = inputOriginal
                tmlPattern = os.path.join(tmlDirName, '*.tml')
                outputFiles = glob.glob(tmlPattern)
                pattern = ".*/(?:"+"|".join(docIDs)+").tml"
                try: 
                    re.compile(pattern)
                except: 
                    raise ValueError(f"{pattern} is not a valid regular expression!")
                outputFiles = [f for f in outputFiles if re.match(pattern, f)]
                if len(outputFiles) == 0:
                    raise RuntimeError(f"{pattern} did not match any file!")
                documents = {}
                for outFile in outputFiles:
                    with open(outFile,"r") as f:
                        curOut = f.read()
                    parser = MyHTMLParser()
                    parser.feed(curOut)
                    curDoc = parser.getDocDict()
                    modDoc = {
                        "docid": curDoc["docid"],
                        "events": {},
                        "eventsByEIID": {},
                        "text": curDoc["text"]
                    } 
                    for e in curDoc["events"]:
                        if e["eid"] not in modDoc["events"].keys():
                            modDoc["events"][e["eid"]] = e["loc"]
                        else: 
                            print("Duplicate eid: {} in doc with docid: {}! This should not have happened!".format(e["eid"], modDoc["docid"]))
                            exit(0)
                    for e in curDoc["instances"]:
                        if e["eiid"] not in modDoc["eventsByEIID"].keys():
                            if e["eid"] in modDoc["events"].keys():
                                modDoc["eventsByEIID"][e["eiid"]] = modDoc["events"][e["eid"]]
                            else: 
                                print("Invalid eid: {} in doc with docid: {}! This should not have happened!".format(e["eid"], modDoc["docid"]))
                                exit(0)
                        else: 
                            print("Duplicate eiid: {} in doc with docid: {}! This should not have happened!".format(e["eiid"], modDoc["docid"]))
                            exit(0)
                    documents[curDoc["docid"]] = modDoc
                passages = {}
                for d in data:
                    if d[0] in documents.keys():
                        doc = documents[d[0]]
                        event1 = "<span style='color:red;'><strong>"+d[1]+" </strong></span>"
                        event2 = "<span style='color:blue;'><strong>"+d[2]+" </strong></span>"
                        loc1 = doc["eventsByEIID"]["ei"+d[3]]
                        loc2 = doc["eventsByEIID"]["ei"+d[4]]
                        if loc1 > loc2:
                            loc1, loc2 = loc2, loc1 
                            event1, event2 = event2, event1
                        modText = doc["text"][:loc1] + event1 + doc["text"][loc1+len(d[1]):loc2] + event2 + doc["text"][loc2+len(d[2]):]
                        if d[0] not in passages.keys():
                            passages[d[0]] = []
                        passages[d[0]].append({
                            "docid": d[0],
                            "bodygraph": modText,
                            "decision": d[5].replace("\n","").strip().lower()
                        })
                    else:
                        print(f"Unable to find document with docID: {d[0]}! This should not have happened!")
                        exit(0) 
                for p in passages.keys():
                    chosenPair = np.random.choice(len(passages[p]))
                    passages[p] = [passages[p][chosenPair]]
                data = list(passages.values())
            else: 
                passages = {}
                for d in data:
                    if d[""] not in passages.keys():
                        passages[d[""]] = {
                            "groupID": d[""],
                            "instances": []
                        }
                    passages[d[""]]["instances"].append(d)
                data = list(passages.values())
        if dataset == "drop":
            if not train: 
                with open(inputOriginal,"r", encoding='utf-8-sig') as f:
                    dataOriginal = json.load(f)
                oriQuestions = {}
                for d in dataOriginal.keys():
                    if d not in oriQuestions.keys():
                        oriQuestions[d] = {}
                    for q in dataOriginal[d]["qa_pairs"]:
                        if q["query_id"] not in oriQuestions[d].keys():
                            oriQuestions[d][q["query_id"]] = q 
                        else: 
                            print("Query ID {} already seen in {}! This should not have happpened!".format(q["query_id"], d))
                for d in data.keys():
                    dOri = d
                    if dOri not in oriQuestions.keys():
                        dOri = "_".join(d.split("_")[:-1])
                        if dOri not in oriQuestions.keys():
                            print("Couldn't find {} in original file!".format(dOri))
                            continue
                    qsToAdd = {}
                    for q in data[d]["qa_pairs"]:
                        qID = "".join(q["query_id"].split("_")[:-1])
                        if qID in oriQuestions[dOri].keys() and qID not in qsToAdd.keys():
                            qsToAdd[qID] = oriQuestions[dOri][qID]
                        elif qID not in qsToAdd.keys(): 
                            print("Could not find original question with qID = {}! This should not have happened!".format(qID))
                    data[d]["qa_pairs"].extend(list(qsToAdd.values()))
            if train:
                newData = {}
                for k  in data.keys():
                    newData[k] = data[k]
                    chosenQue = np.random.choice(len(data[k]["qa_pairs"]))
                    newData[k]["qa_pairs"] = [data[k]["qa_pairs"][chosenQue]]
                data = newData
            data = list(data.values())
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