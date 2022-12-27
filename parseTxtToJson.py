import argparse
import json
import sys
from os.path import exists
import warnings

parser = argparse.ArgumentParser()

parser.add_argument(
    "-input",
    help="Path to input txt file",
    required=True,
)

parser.add_argument(
    "-output",
    help="Path to output json file",
    required=True,
)

parser.add_argument(
    "-inputID",
    help="ID to identify input file; This number would be appended to all passageIDs",
    required=True,
)

args = parser.parse_args()
inputFile = args.input
outputFile = None
if args.output:
    outputFile = args.output
inputID = args.inputID
#------------------------------------
def parse(a):
    examples = []
    i = 0
    while(i < len(a) and "PassageID: " in a[i]):
        passageID = a[i][len("PassageID: "):].strip()
        i += 1
        passage = a[i][len("Passage: "):].strip()
        i += 1
        #Skip \n
        i += 1
        #Skip Questions
        i += 1
        while("QuestionID: " in a[i]):
            questionID = a[i][len("QuestionID: "):].strip()
            i += 1
            question = a[i][len("Question: "):].strip()
            i += 1
            answer = a[i][len("Answer: "):].strip()
            i += 1
            explanation = a[i][len("Explanation: "):].strip()
            #CondaQA format
            examples.append({
                "PassageID": str(inputID) + "_" + str(passageID),
                "sentence1": passage,
                "QuestionID": questionID,
                "sentence2": question,
                "label": answer,
                "explanation": explanation
            })
            i += 1
            #Skip \n
            i += 1
        #Skip --------------------------------------------------
        i += 1
    print(len(examples))
    return examples
#------------------------------------
def main():
    if not inputFile.endswith(".txt") or not exists(inputFile):
        raise Exception("-input argmument invalid!")
    if not outputFile.endswith(".json"):
        raise Exception("-output argmument invalid!")
    with open(inputFile, "r") as f:
        dataInFile = f.readlines()
        examples = parse(dataInFile)
        with open(outputFile,"w") as of:
            json.dump(examples, of, indent=4)
#------------------------------------
if __name__ == "__main__":
    main()