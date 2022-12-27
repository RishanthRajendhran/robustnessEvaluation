#!/bin/bash
#SBATCH --account marasovic-gpu-np
#SBATCH --partition marasovic-gpu-np
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH -o outputs-%j

ISTRAINDIR=false
ISTESTDIR=false

while getopts 'a:b:c:d:efg:' opt; do
  case "$opt" in
    a)   PROMPTTYPE="$OPTARG"   ;;
    b)    BESTPROMPTTYPE="$OPTARG"   ;;
    c) TRAIN="$OPTARG" ;;
    d)   TEST="$OPTARG"   ;;
    e)   ISTRAINDIR=true     ;;
    f)    ISTESTDIR=true     ;;
    g)  DATASET="$OPTARG"   ;;
    *) echo "Unexpected option: $1 - this should not happen."
       usage ;;
  esac
done

if [ -z ${PROMPTTYPE+x} ]; then 
    echo "promptType (-a) not specified"; 
    exit 1; 
fi

if [ -z ${BESTPROMPTTYPE+x} ]; then 
    echo "bestPromptType (-b) not specified"; 
    exit 1; 
fi

if [ -z ${TRAIN+x} ]; then 
    echo "train (-c) not specified"; 
    exit 1; 
fi

if [ -z ${TEST+x} ]; then 
    echo "TEST (-d) not specified"; 
    exit 1; 
fi

if [ -z ${DATASET+x} ]; then 
    echo "DATASET (-g) not specified"; 
    exit 1; 
fi

# echo "promptType = $PROMPTTYPE"
# echo "bestPromptType = $BESTPROMPTTYPE"
# echo "train = $TRAIN"
# echo "test = $TEST"
# echo "isTrainDir = $ISTRAINDIR"
# echo "isTestDir = $ISTESTDIR"
# echo "Dataset = $DATASET"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate flant5Env

# wandb disabled 
export TRANSFORMER_CACHE="/scratch/general/vast/u1419542/huggingface_cache"
if [ "$ISTRAINDIR" = true ] ; then 
    if [ "$ISTESTDIR" = true ] ; then 
        python3.9 test.py -dataset $DATASET -promptType $PROMPTTYPE -bestPromptType $BESTPROMPTTYPE -train $TRAIN -test $TEST -isTrainDir -isTestDir ;
    else 
        python3.9 test.py -dataset $DATASET -promptType $PROMPTTYPE -bestPromptType $BESTPROMPTTYPE -train $TRAIN -test $TEST -isTrainDir ;
    fi ;
else 
    if [ "$ISTESTDIR" = true ] ; then 
        python3.9 test.py -dataset $DATASET -promptType $PROMPTTYPE -bestPromptType $BESTPROMPTTYPE -train $TRAIN -test $TEST -isTestDir ;
    else 
        python3.9 test.py -dataset $DATASET -promptType $PROMPTTYPE -bestPromptType $BESTPROMPTTYPE -train $TRAIN -test $TEST ;
    fi ;
fi