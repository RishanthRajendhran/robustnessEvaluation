import argparse
import json
import sys
from os.path import exists
import re
import nltk
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument(
    "-dataset",
    choices=["boolq", 
            "ropes", 
            "drop", 
            "mctaco", 
            "quoref", 
            "imdb", 
            "matres", 
            "nlvr2",
            "perspectrum",
            "udparsing",
            ],
    required=True,
    help="Name of dataset",
)

parser.add_argument(
    "-input",
    help="Path to input json file",
    required=True,
)

parser.add_argument(
    "-output",
    help="Path to output txt file; Default: standard output"
)

parser.add_argument(
    "-train",
    action="store_true",
    help="Boolean flag to indicate that input is train set [Useful for Perspectrum/UDParsing only]"
)

args = parser.parse_args()
dataset = args.dataset
inputFile = args.input
outputFile = None
train = args.train
if args.output:
    outputFile = args.output

#------------------------------------
def prettyPrint(a, dataset):
    if dataset == "boolq":
        for i in range(len(a)):
            print(f"PassageID: {i}")
            p = a[i]
            #Only present in contrastive set; train set calls it passage
            if "paragraph" in p.keys():
                passage = p["paragraph"]
            else:
                passage = p["passage"]
            print(f"Passage: {passage}",end="")
            print("\n")
            print("Questions:")

            print(f"QuestionID: 0")

            que = p["question"]
            print(f"Question: {que}")

            ans = p["answer"]
            print(f"Answer: {ans}")

            exp = ""
            if "explanation" in p.keys():
                exp = p["explanation"]
            print(f"Explanation: {exp}",end="")
            print("\n")

            #Only present for contrastive set
            if "perturbed_questions" in p.keys():
                for q in range(len(p["perturbed_questions"])):
                    print(f"QuestionID: 0")

                    que = p["perturbed_questions"][q]["perturbed_q"]  
                    print(f"Question: {que}")

                    ans = p["perturbed_questions"][q]["answer"]
                    print(f"Answer: {ans}")

                    exp = ""
                    if "explanation" in p["perturbed_questions"][q].keys():
                        exp = p["perturbed_questions"][q]["explanation"]
                    print(f"Explanation: {exp}",end="")
                    print("\n")
            print("-"*50)
    elif dataset == "ropes":
        for i in range(len(a)):
            print(f"PassageID: {i}")
            p = a[i]
            passage = p["background"]
            passage = passage.strip()
            passage = passage.replace("\n", " ")
            print(f"Passage: {passage}",end="")
            print("\n")
            print("Questions:")
            for q in range(len(p["qas"])):
                qID = p["qas"][q]["id"]
                print(f"QuestionID: {qID}")

                # que = p["situation"].replace("\n", " ") + " " + p["qas"][q]["question"]  
                if type(p["qas"][q]) is dict and "situation" in p["qas"][q].keys(): 
                    que = p["qas"][q]["situation"].replace("\n", " ") + " " + p["qas"][q]["question"].replace("\n", " ")  
                else:
                    que = p["situation"].replace("\n", " ") + " " + p["qas"][q]["question"].replace("\n", " ")  
                print(f"Question: {que}")

                ansList = []
                for k in range(len(p["qas"][q]["answers"])):
                    for l in p["qas"][q]["answers"][k].keys():
                        ansList.append(p["qas"][q]["answers"][k][l])
                ans = ", ".join(ansList)
                print(f"Answer: {ans}")

                exp = ""
                if "explanation" in p["qas"][q].keys():
                    exp = p["qas"][q]["explanation"]
                print(f"Explanation: {exp}",end="")
                print("\n")
            print("-"*50)
    elif dataset == "drop":
        for i in range(len(a)):
            print(f"PassageID: {i}")
            p = a[i]
            passage = p["passage"]
            print(f"Passage: {passage}",end="")
            print("\n")
            print("Questions:")
            for q in range(len(p["qa_pairs"])):
                qID = p["qa_pairs"][q]["query_id"].split("_")[0]
                print(f"QuestionID: {qID}")

                que = p["qa_pairs"][q]["question"]  
                print(f"Question: {que}")

                ansList = []
                number = p["qa_pairs"][q]["answer"]["number"]
                dateDay = p["qa_pairs"][q]["answer"]["date"]["day"]
                dateMonth = p["qa_pairs"][q]["answer"]["date"]["month"]
                dateYear = p["qa_pairs"][q]["answer"]["date"]["year"]
                spans = ", ".join(p["qa_pairs"][q]["answer"]["spans"])
                if number != "":
                    ansList.append(number)
                if dateDay != "":
                    ansList.append(dateDay)
                if dateMonth != "":
                    ansList.append(dateMonth)
                if dateYear != "":
                    ansList.append(dateYear)
                if spans != "":
                    ansList.append(spans)
                ans = ", ".join(ansList)
                print(f"Answer: {ans}")

                exp = ""
                if "explanation" in p["qa_pairs"][q].keys():
                    exp = p["qa_pairs"][q]["explanation"]
                elif "explanation" in p["qa_pairs"][q]["answer"].keys():
                    exp = p["qa_pairs"][q]["answer"]["explanation"]
                print(f"Explanation: {exp}",end="")
                print("\n")
            print("-"*50)
    elif dataset == "mctaco":
        for i in range(len(a)):
            print(f"PassageID: {i}")
            p = a[i]
            passage = p["sentence1"]
            print(f"Passage: {passage}",end="")
            print("\n")
            print("Questions:")
            for q in range(len(p["qas"])):
                print(f"QuestionID: {q}")

                que = p["qas"][q]["sentence2"]  
                print(f"Question: Is this the answer: {que}?")

                ans = p["qas"][q]["label"]
                print(f"Answer: {ans}")

                exp = ""
                if "explanation" in p["qas"][q].keys():
                    exp = p["qas"][q]["explanation"]
                print(f"Explanation: {exp}",end="")
                print("\n")
            print("-"*50)
    elif dataset == "quoref":
        for i in range(len(a)):
            print(f"PassageID: {i}")
            p = a[i]
            passage = p["context"]
            passage = passage.replace("\n", " ")
            print(f"Passage: {passage}",end="")
            print("\n")
            print("Questions:")
            for q in range(len(p["qas"])):
                if "original_id" in p["qas"][q].keys():
                    qID = p["qas"][q]["original_id"]
                else: 
                    qID = p["qas"][q]["id"]
                print(f"QuestionID: {qID}")

                que = p["qas"][q]["question"]  
                print(f"Question: {que}")

                ansList = []
                for k in range(len(p["qas"][q]["answers"])):
                    ansList.append(p["qas"][q]["answers"][k]["text"])
                ans = ", ".join(ansList)
                print(f"Answer: {ans}")

                exp = ""
                if "explanation" in p["qas"][q].keys():
                    exp = p["qas"][q]["explanation"]
                print(f"Explanation: {exp}",end="")
                print("\n")
            print("-"*50)
    elif dataset == "imdb":
        for i in range(len(a)):
            for p in a[i]:
                print(f"PassageID: {i}")
                passage = p["passage"]
                print(f"Passage: {passage}",end="")
                print("\n")
                print("Questions:")

                for q in range(len(p["qas"])):
                    print(f"QuestionID: {q}")

                    que = p["qas"][q]["question"]  
                    que = que.strip()
                    print(f"Question: {que}")

                    ans = p["qas"][q]["answer"]
                    print(f"Answer: {ans}")

                    exp = ""
                    if "explanation" in p["qas"][q].keys():
                        exp = p["qas"][q]["explanation"]
                    print(f"Explanation: {exp}",end="")
                    print("\n")
                print("-"*50)
    elif dataset == "matres":
        if train: 
            for i in range(len(a)):
                print(f"PassageID: {i}")
                p = a[i][0]

                #bodygraph
                if "bodygraph" not in p.keys():
                    raise RuntimeError("\"bodygraph\" not a key!")
                que = p["bodygraph"]
                #Events extraction
                firstEventPatt = "<span style='color:(red|blue);'>.*?</span>"
                firstEventMatch = re.search(firstEventPatt, que)
                if firstEventMatch == None:
                    raise RuntimeError("re.search returned None!")
                firstEventMatch = firstEventMatch.group()
                firstEvent = re.search("<strong>.*?</strong>", firstEventMatch)
                if firstEvent == None:
                    raise RuntimeError("re.search returned None!")
                firstEvent = firstEvent.group()
                firstEvent = firstEvent[len("<strong>"):-(len("</strong>"))].strip()
                firstEventRepl = " *"+firstEvent+"* "
                que = re.sub(firstEventPatt, firstEventRepl, que, count=1)

                secEventPatt = "<span style='color:(red|blue);'>.*?</span>"
                secEventMatch = re.search(secEventPatt, que)
                if secEventMatch == None:
                    raise RuntimeError("re.search returned None!")
                secEventMatch = secEventMatch.group()
                secEvent = re.search("<strong>.*?</strong>", secEventMatch)
                if secEvent == None:
                    raise RuntimeError("re.search returned None!")
                secEvent = secEvent.group()
                secEvent = secEvent[len("<strong>"):-(len("</strong>"))].strip()
                secEventRepl = " *"+secEvent+"* "
                que = re.sub(secEventPatt, secEventRepl, que, count=1)

                tagMatch = re.search("<.*?>", que)
                while tagMatch:
                    que = re.sub("<.*?>","",que)
                    tagMatch = re.search("<.*?>", que)
                que = que.strip()

                passage = que.replace("\n", "").strip()
                print(f"Passage: {passage}",end="")
                print("\n")
                print("Questions:")

                print(f"QuestionID: 0")
                que = f"When did the event *{firstEvent}* happen in relation to the event *{secEvent}*: before, after, simultaneously, or is it vague?"
                print(f"Question: {que}")
                ans = p["decision"]
                print(f"Answer: {ans}")
                exp = ""
                if "explanation" in p.keys():
                    exp = p["explanation"]
                print(f"Explanation: {exp}",end="")
                print("\n")
                print("-"*50)
        else:
            seenQuestions = []
            for i in range(len(a)):
                for j in range(len(a[i]["instances"])):
                    p = a[i]["instances"][j]
                    #bodygraph
                    if "bodygraph" not in p.keys():
                        raise RuntimeError("\"bodygraph\" not a key!")
                    
                    que = p["bodygraph"]
                    if len(que) and que not in seenQuestions:
                        seenQuestions.append(que)
                        #Events extraction
                        firstEventPatt = "<span style='color:(red|blue);'>.*?</span>"
                        firstEventMatch = re.search(firstEventPatt, que)
                        if firstEventMatch == None:
                            raise RuntimeError("re.search returned None!")
                        firstEventMatch = firstEventMatch.group()
                        firstEvent = re.search("<strong>.*?</strong>", firstEventMatch)
                        if firstEvent == None:
                            raise RuntimeError("re.search returned None!")
                        firstEvent = firstEvent.group()
                        firstEvent = firstEvent[len("<strong>"):-(len("</strong>"))].strip()
                        firstEventRepl = " *"+firstEvent+"* "
                        que = re.sub(firstEventPatt, firstEventRepl, que, count=1)

                        secEventPatt = "<span style='color:(red|blue);'>.*?</span>"
                        secEventMatch = re.search(secEventPatt, que)
                        if secEventMatch == None:
                            raise RuntimeError("re.search returned None!")
                        secEventMatch = secEventMatch.group()
                        secEvent = re.search("<strong>.*?</strong>", secEventMatch)
                        if secEvent == None:
                            raise RuntimeError("re.search returned None!")
                        secEvent = secEvent.group()
                        secEvent = secEvent[len("<strong>"):-(len("</strong>"))].strip()
                        secEventRepl = " *"+secEvent+"* "
                        que = re.sub(secEventPatt, secEventRepl, que, count=1)

                        tagMatch = re.search("<.*?>", que)
                        while tagMatch:
                            que = re.sub("<.*?>","",que)
                            tagMatch = re.search("<.*?>", que)
                        que = que.strip()

                        print("PassageID: {}".format(a[i]["groupID"]))
                        passage = que.replace("\n","")
                        print(f"Passage: {passage}",end="")
                        print("\n")
                        print("Questions:")

                        print(f"QuestionID: {2*j}")
                        que = f"When did the event *{firstEvent}* happen in relation to the event *{secEvent}*: before, after, simultaneously, or is it vague?"
                        print(f"Question: {que}")
                        ans = p["decision"]
                        print(f"Answer: {ans}")
                        exp = ""
                        if "explanation" in p.keys():
                            exp = p["explanation"]
                        print(f"Explanation: {exp}",end="")
                        print("\n")
                        print("-"*50)

                    #modified bodygraph
                    if "modified bodygraph" not in p.keys():
                        raise RuntimeError("\"modified bodygraph\" not a key!")
                    que = p["modified bodygraph"]
                    if len(que) and que not in seenQuestions:
                        seenQuestions.append(que)
                        #Events extraction
                        firstEventPatt = "<span style='color:(red|blue);'>.*?</span>"
                        firstEventMatch = re.search(firstEventPatt, que)
                        if firstEventMatch == None:
                            raise RuntimeError(f"re.search in {que} returned None!")
                        firstEventMatch = firstEventMatch.group()
                        firstEvent = re.search("<strong>.*?</strong>", firstEventMatch)
                        if firstEvent == None:
                            raise RuntimeError(f"re.search in {que} returned None!")
                        firstEvent = firstEvent.group()
                        firstEvent = firstEvent[len("<strong>"):-(len("</strong>"))].strip()
                        firstEventRepl = " *"+firstEvent+"* "
                        que = re.sub(firstEventPatt, firstEventRepl, que, count=1)

                        secEventPatt = "<span style='color:(red|blue);'>.*?</span>"
                        secEventMatch = re.search(secEventPatt, que)
                        if secEventMatch == None:
                            raise RuntimeError(f"re.search in {que} returned None!")
                        secEventMatch = secEventMatch.group()
                        secEvent = re.search("<strong>.*?</strong>", secEventMatch)
                        if secEvent == None:
                            raise RuntimeError(f"re.search in {que} returned None!")
                        secEvent = secEvent.group()
                        secEvent = secEvent[len("<strong>"):-(len("</strong>"))].strip()
                        secEventRepl = " *"+secEvent+"* "
                        que = re.sub(secEventPatt, secEventRepl, que, count=1)

                        tagMatch = re.search("<.*?>", que)
                        while tagMatch:
                            que = re.sub("<.*?>","",que)
                            tagMatch = re.search("<.*?>", que)
                        que = que.strip()

                        print("PassageID: {}".format(a[i]["groupID"]))
                        passage = que.replace("\n","")
                        print(f"Passage: {passage}",end="")
                        print("\n")
                        print("Questions:")

                        print(f"QuestionID: {2*j+1}")
                        que = f"When did the event *{firstEvent}* happen in relation to the event *{secEvent}*: before, after, simultaneously, or is it vague?"
                        print(f"Question: {que}")
                        ans = p["new decision"]
                        print(f"Answer: {ans}")
                        exp = ""
                        if "explanation" in p.keys():
                            exp = p["explanation"]
                        print(f"Explanation: {exp}",end="")
                        print("\n")
                        print("-"*50)
    elif dataset == "nlvr2":
        pass
        # for i in range(len(a)):
        #     print(f"PassageID: {i}")
        #     p = a[i]
        #     passage = "Consider the following two images: "
        #     passage += p["left_url"]
        #     passage += " and "
        #     passage += p["right_url"]
        #     passage += " ."
        #     print(f"Passage: {passage}",end="")
        #     print("\n")
        #     print("Questions:")

        #     print(f"QuestionID: 0")

        #     que = p["qas"][q]["question"]  
        #     que = que.strip()
        #     queAttach = " What is the sentiment of this review: Positive or Negative?"
        #     if que[-1] != ".":
        #         queAttach = "." + queAttach
        #     print(f"Question: {que}{queAttach}")

        #     ans = p["qas"][q]["answer"]
        #     print(f"Answer: {ans}")

        #     exp = ""
        #     if "explanation" in p["qas"][q].keys():
        #         exp = p["qas"][q]["explanation"]
        #     print(f"Explanation: {exp}",end="")
        #     print("\n")
        #     print("-"*50)
    elif dataset == "perspectrum":
        if train: 
            for i in range(len(a)):
                p = a[i]
                for q in range(len(p["perspectives"])):
                    print(f"PassageID: {i}")
                    passage = p["perspectives"][q]["perspective"]["text"]
                    passage = passage.replace("\n","").strip()
                    print(f"Passage: Perspective: {passage}",end="")
                    print("\n")
                    print("Questions:")

                    print(f"QuestionID: {q}")
                    que = "Claim: " +  p["claim"].strip()
                    if que[-1] != ".":
                        que += "."
                    queAttach = " Does the perspective support or undermine the claim? Answer with: supports, undermines or not a valid perspective."
                    print(f"Question: {que}{queAttach}")

                    ans = p["perspectives"][q]["stance_label"]
                    if ans == "pos":
                        ans = "supports"
                    elif ans == "neg":
                        ans = "undermines"
                    elif ans == "unk":
                        ans = "not a valid perspective"
                    print(f"Answer: {ans}")

                    exp = ""
                    if "explanation1" in p["perspectives"][q].keys():
                        exp = p["qas"][q]["explanation1"]
                    print(f"Explanation: {exp}",end="")
                    print("\n")
                    print("-"*50)
        else:
            for i in range(len(a)):
                p = a[i]
                for q in range(len(p["perspectives"])):
                    print(f"PassageID: {i}")
            
                    passage = p["perspectives"][q]["perspective"]
                    passage = passage.replace("\n","").strip()
                    print(f"Passage: Perspective: {passage}",end="")
                    print("\n")
                    print("Questions:")
                
                    print(f"QuestionID: {q}")
                    que = "Claim: " +  p["original_claim"].strip()
                    if que[-1] != ".":
                        que += "."
                    queAttach = " Does the perspective support or undermine the claim? Answer with: supports, undermines or not a valid perspective."
                    print(f"Question: {que}{queAttach}")

                    ans = p["perspectives"][q]["original_stance_label"]
                    if ans == "pos":
                        ans = "supports"
                    elif ans == "neg":
                        ans = "undermines"
                    elif ans == "unk":
                        ans = "not a valid perspective"
                    print(f"Answer: {ans}")

                    exp = ""
                    if "explanation1" in p["perspectives"][q].keys():
                        exp = p["qas"][q]["explanation1"]
                    print(f"Explanation: {exp}",end="")
                    print("\n")

                    print(f"QuestionID: {q}")
                    que = "Claim: " +  p["contrast_claim"].strip()
                    if que[-1] != ".":
                        que += "."
                    queAttach = " Does the perspective support or undermine the claim? Answer with: supports, undermines or not a valid perspective."
                    print(f"Question: {que}{queAttach}")

                    ans = p["perspectives"][q]["contrast_stance_label"]
                    if ans == "pos":
                        ans = "supports"
                    elif ans == "neg":
                        ans = "undermines"
                    elif ans == "unk":
                        ans = "not a valid perspective"
                    print(f"Answer: {ans}")

                    exp = ""
                    if "explanation2" in p["perspectives"][q].keys():
                        exp = p["qas"][q]["explanation2"]
                    print(f"Explanation: {exp}",end="")
                    print("\n")
                    print("-"*50)
    elif dataset == "udparsing":
        if train:
            for i in range(len(a)):
                p = a[i]
                sentID = p["sent_id"]
                tokensByID = {}
                ADPids = []
                for t in range(len(p["tokens"])):
                    token = p["tokens"][t]
                    if type(token["id"]) is list:
                        continue
                    if token["id"] in tokensByID.keys():
                        print("Token with ID = {} already seen!".format(token["id"]))
                        exit(0)
                    tokensByID[token["id"]] = token
                    if token["upos"] == "ADP":
                        ADPids.append(token["id"])
                ADPids = [ADPids[np.random.choice(len(ADPids))]]
                for ADPid in ADPids:
                    ADPnode = tokensByID[ADPid]
                    if ADPnode["head"] == 0:
                        parent = ADPnode
                    else:
                        parent =  tokensByID[ADPnode["head"]]
                    if parent["head"] == 0:
                        grandparent = parent
                    else:
                        grandparent = tokensByID[parent["head"]]

                    passage = ""
                    for token in p["tokens"]:
                        if type(token["id"]) is list: 
                            continue
                        if not ((token["upos"] == "PUNCT") or (token["upos"] == "PART" and (token["xpos"] == "POS" or token["xpos"] == "RB"))):
                            passage += " "
                        if token["id"] == ADPid:
                            passage += "*"
                        passage += token["form"]
                        if token["id"] == ADPid:
                            passage += "*"

                    print(f"PassageID: {sentID}")
                    # wordsInPassage = nltk.tokenize.wordpunct_tokenize(passage)
                    passage = passage.replace("\n", "")
                    passage = passage.strip()
                    print(f"Passage: Sentence: {passage}",end="")
                    print("\n")
                    print("Questions:")

                    que = "What is the parent word of the parent of the preposition *{}* in the sentence?".format(ADPnode["form"])
                    que = que.replace("\n","")
                    que = que.strip()
                    print("QuestionID: {}".format(ADPids.index(ADPid)))
                    print(f"Question: {que}")

                    ans = grandparent["form"]
                    print(f"Answer: {ans}")

                    exp = ""
                    if "explanation" in p.keys():
                        exp = p["explanation"]
                    print(f"Explanation: {exp}",end="")
                    print("\n")
                    print("-"*50)
        else: 
            for i in range(len(a)):
                for j in range(len(a[i])):
                    p = a[i][j]
                    sentID = p["sent_id"]
                    tokensByID = {}
                    # ADPids = []
                    for t in range(len(p["tokens"])):
                        token = p["tokens"][t]
                        if type(token["id"]) is list:
                            continue
                        if token["id"] in tokensByID.keys():
                            print("Token with ID = {} already seen!".format(token["id"]))
                            print("Problematic sentence: {}".format(sentID))
                            print("Problematic token: {}".format(token))
                            exit(0)
                        tokensByID[token["id"]] = token
                        # if token["upos"] == "ADP":
                        #     ADPids.append(token["id"])
                    parentADPid = p["parentADPid"]
                    ADPid = p["ADPid"]
                    ADPnode = tokensByID[ADPid]
                    if ADPnode["head"] == 0:
                        parent = ADPnode
                    else:
                        parent =  tokensByID[ADPnode["head"]]
                    if parent["head"] == 0:
                        grandparent = parent
                    else:
                        grandparent = tokensByID[parent["head"]]

                    passage = ""
                    for token in p["tokens"]:
                        if type(token["id"]) is list: 
                            continue
                        if not ((token["upos"] == "PUNCT") or (token["upos"] == "PART" and (token["xpos"] == "POS" or token["xpos"] == "RB"))):
                            passage += " "
                        if token["id"] == ADPid:
                            passage += "*"
                        passage += token["form"]
                        if token["id"] == ADPid:
                            passage += "*"

                    print(f"PassageID: {sentID}_{parentADPid}")
                    # wordsInPassage = nltk.tokenize.wordpunct_tokenize(passage)
                    passage = passage.replace("\n", "")
                    passage = passage.strip()
                    print(f"Passage: Sentence: {passage}",end="")
                    print("\n")
                    print("Questions:")

                    que = "What is the parent word of the parent of the preposition *{}* in the sentence?".format(ADPnode["form"])
                    que = que.replace("\n","")
                    que = que.strip()
                    print("QuestionID: {}".format(0))
                    print(f"Question: {que}")

                    ans = grandparent["form"]
                    print(f"Answer: {ans}")

                    exp = ""
                    if "explanation" in p.keys():
                        exp = p["explanation"]
                    print(f"Explanation: {exp}",end="")
                    print("\n")
                    print("-"*50)
    else:
        print(f"Unrecognized dataset!")
#------------------------------------
def main():
    if not inputFile.endswith(".json") or not exists(inputFile):
        raise Exception("-input argmument invalid!")
    if outputFile and not outputFile.endswith(".txt"):
        raise Exception("-output argmument invalid!")
    with open(inputFile, "r") as f:
        dataInFile = json.load(f)
        if outputFile != None:
            sys.stdout = open(outputFile, 'w')
        prettyPrint(dataInFile, dataset)
#------------------------------------
if __name__ == "__main__":
    main()