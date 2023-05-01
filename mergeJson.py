import argparse
import json
import sys
from os.path import exists, join
from glob import glob
import re

parser = argparse.ArgumentParser()

parser.add_argument(
    "-input",
    nargs="+",
    help="Path to input json files to merge",
    required=True,
)

parser.add_argument(
    "-output",
    help="Path to merged output json file",
    required=True,
)

parser.add_argument(
    "-isDir",
    action="store_true",
    help="Boolean flag to indicate if -input is the path to the directory containing files to be merged"
)

parser.add_argument(
    "-pattern",
    help="RegEx pattern for json file names in the directory that need to be merged"
)

parser.add_argument(
    "-printInfo",
    action="store_true",
    help="Print information about files being merged"
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
                "perspectrum"
            ],
    required=True
)

args = parser.parse_args()
inputFiles = args.input
outputFile = None
isDir = False 
pattern = None
printInfo = args.printInfo
dataset = args.dataset
if args.output:
    outputFile = args.output
if args.isDir:
    isDir = args.isDir
if args.pattern:
    pattern = args.pattern

if isDir:
    jsonDirName = inputFiles[0]
    jsonPattern = join(jsonDirName, '*.json')
    inputFiles = glob(jsonPattern)
    if pattern:
        try: 
            re.compile(pattern)
        except: 
            raise ValueError(f"{pattern} is not a valid regular expression!")
        inputFiles = [f for f in inputFiles if re.match(pattern, f)]
        if len(inputFiles) == 0:
            raise RuntimeError(f"{pattern} did not match any file!")
#------------------------------------
def main():
    for f in inputFiles:
        if not f.endswith(".json") or not exists(f):
            raise Exception("-input argmument invalid!")
    if not outputFile.endswith(".json"):
        raise Exception("-output argmument invalid!")
    examples = []
    for f in inputFiles:
        if dataset == "condaqa":
            data = []
            for  line in open(f, "r"):
                data.append(json.loads(line))
            examples.extend(data)
        else:
            with open(f,"r", encoding='utf-8-sig') as rf:
                examples.extend(json.load(rf))
    if dataset == "condaqa":
        with open(outputFile,"w") as of:
            for ex in examples:
                json.dump(ex, of)
                of.write("\n")
    else:
        with open(outputFile,"w") as of:
            json.dump(examples, of, indent=4)
    if printInfo:
        print(f"Merged the following files:")
        for i in inputFiles:
            print(f"\t{i}")
        print(f"Merged file: {outputFile}")
#------------------------------------
if __name__ == "__main__":
    main()