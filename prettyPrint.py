import argparse
import json
import sys
from os.path import exists
import re

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

args = parser.parse_args()
dataset = args.dataset
inputFile = args.input
outputFile = None
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
                que = p["qas"][q]["situation"].replace("\n", " ") + " " + p["qas"][q]["question"]  
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
                print(f"QuestionID: {q}")

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
            print(f"PassageID: {i}")
            p = a[i]
            passage = p["passage"]
            print(f"Passage: {passage}",end="")
            print("\n")
            print("Questions:")

            for q in range(len(p["qas"])):
                print(f"QuestionID: {q}")

                que = p["qas"][q]["question"]  
                que = que.strip()
                queAttach = " What is the sentiment of this review: Positive or Negative?"
                if que[-1] != ".":
                    queAttach = "." + queAttach
                print(f"Question: {que}{queAttach}")

                ans = p["qas"][q]["answer"]
                print(f"Answer: {ans}")

                exp = ""
                if "explanation" in p["qas"][q].keys():
                    exp = p["qas"][q]["explanation"]
                print(f"Explanation: {exp}",end="")
                print("\n")
            print("-"*50)
    elif dataset == "matres":
        for i in range(len(a)):
            print(f"PassageID: {i}")
            p = a[i]

            passage = ""
            print(f"Passage: {passage}",end="")
            print("\n")
            print("Questions:")
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
            firstEvent = firstEvent[len("<strong>"):-(len("</strong>"))]
            firstEventRepl = " (("+firstEvent.strip()+")) "
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
            secEvent = secEvent[len("<strong>"):-(len("</strong>"))]
            secEventRepl = " [["+secEvent.strip()+"]] "
            que = re.sub(secEventPatt, secEventRepl, que, count=1)

            tagMatch = re.search("<.*?>", que)
            while tagMatch:
                que = re.sub("<.*?>","",que)
                tagMatch = re.search("<.*?>", que)
            que = que.strip()

            print(f"QuestionID: 0")
            queAttach = " When did the event within double soft parentheses i.e. ((.)) happen with respect to the event within double square brackets i.e. [[.]]: before/after/simultaneous/vague?"
            if que[-1] != ".":
                queAttach = "." + queAttach
            print(f"Question: {que}{queAttach}")
            ans = p["decision"]
            print(f"Answer: {ans}")
            exp = ""
            if "explanation" in p.keys():
                exp = p["explanation"]
            print(f"Explanation: {exp}",end="")
            print("\n")

            #modified bodygraph
            if "modified bodygraph" not in p.keys():
                raise RuntimeError("\"modified bodygraph\" not a key!")
            que = p["modified bodygraph"]
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
            firstEvent = firstEvent[len("<strong>"):-(len("</strong>"))]
            firstEventRepl = " (("+firstEvent.strip()+")) "
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
            secEvent = secEvent[len("<strong>"):-(len("</strong>"))]
            secEventRepl = " [["+secEvent.strip()+"]] "
            que = re.sub(secEventPatt, secEventRepl, que, count=1)

            tagMatch = re.search("<.*?>", que)
            while tagMatch:
                que = re.sub("<.*?>","",que)
                tagMatch = re.search("<.*?>", que)
            que = que.strip()

            print(f"QuestionID: 1")
            queAttach = " When did the event within double soft parentheses i.e. ((.)) happen with respect to the event within double square brackets i.e. [[.]]: before/after/simultaneous/vague?"
            if que[-1] != ".":
                queAttach = "." + queAttach
            print(f"Question: {que}{queAttach}")
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
        for i in range(len(a)):
            print(f"PassageID: {i}")
            p = a[i]
            passage = "Consider the following two claims: "
            passage += "Claim 1: " + p["original_claim"]
            passage += " and "
            passage += "Claim2: " + p["contrast_claim"]
            if passage.strip()[-1] != ".":
                passage += " ."
            print(f"Passage: {passage}",end="")
            print("\n")
            print("Questions:")

            for q in range(len(p["perspectives"])):
                print(f"QuestionID: {q}")
                que = p["perspectives"][q]["perspective"]  
                que = que.strip()
                queAttach = " What is the stance taken by Claim 1 with respect to this perspective: pos/neg/unk ?"
                if que.strip()[-1] != ".":
                    queAttach = "." + queAttach
                print(f"Question: Perspective: {que}{queAttach}")

                ans = p["perspectives"][q]["original_stance_label"]
                print(f"Answer: {ans}")

                exp = ""
                if "explanation1" in p["perspectives"][q].keys():
                    exp = p["qas"][q]["explanation1"]
                print(f"Explanation: {exp}",end="")
                print("\n")

                print(f"QuestionID: {q}")
                queAttach = " What is the stance taken by Claim 2 with respect to this perspective: pos/neg/unk ?"
                if que.strip()[-1] != ".":
                    queAttach = "." + queAttach
                print(f"Question: Perspective: {que}{queAttach}")

                ans = p["perspectives"][q]["contrast_stance_label"]
                print(f"Answer: {ans}")

                exp = ""
                if "explanation2" in p["perspectives"][q].keys():
                    exp = p["qas"][q]["explanation2"]
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