#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --mem=128GB
#SBATCH -o outputs-%j

ISTRAINDIR=false
ISTESTDIR=false
MODELSIZE="xxl"
SELFCONSISTENCY=false
TRAINPATT=".*/.*\\.json"
TESTPATT=".*/.*\\.json"
NOCOT=false

while getopts 'a:b:c:d:efg:h:ij:k:l' opt; do
  case "$opt" in
    a)   PROMPTTYPE="$OPTARG"   ;;
    b)    BESTPROMPTTYPE="$OPTARG"   ;;
    c) TRAIN="$OPTARG" ;;
    d)   TEST="$OPTARG"   ;;
    e)   ISTRAINDIR=true     ;;
    f)    ISTESTDIR=true     ;;
    g)  DATASET="$OPTARG"   ;;
    h)  MODELSIZE="$OPTARG"   ;;
    i)   SELFCONSISTENCY=true     ;;
    j)   TRAINPATT="$OPTARG"     ;;
    k)   TESTPATT="$OPTARG"     ;;
    l)   NOCOT=true     ;;
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

export PYTHONPATH=/scratch/general/vast/u1419542/miniconda3/envs/flant5Env/bin/python
source /scratch/general/vast/u1419542/miniconda3/etc/profile.d/conda.sh
conda activate flant5Env

# wandb disabled 
# mkdir /scratch/general/vast/u1419542/huggingface_cache
export TRANSFORMERS_CACHE="/scratch/general/vast/u1419542/huggingface_cache"
if [ "$ISTRAINDIR" = true ] ; then 
    if [ "$ISTESTDIR" = true ] ; then
        if [ "$SELFCONSISTENCY" = true ] ; then
            if [ "$NOCOT" = true ] ; then 
                python3 test.py -noCoT -selfConsistency -modelSize $MODELSIZE -dataset $DATASET -promptType $PROMPTTYPE -bestPromptType $BESTPROMPTTYPE -train $TRAIN -test $TEST -isTrainDir -isTestDir -trainPattern $TRAINPATT -testPattern $TESTPATT;
            else 
                python3 test.py -selfConsistency -modelSize $MODELSIZE -dataset $DATASET -promptType $PROMPTTYPE -bestPromptType $BESTPROMPTTYPE -train $TRAIN -test $TEST -isTrainDir -isTestDir -trainPattern $TRAINPATT -testPattern $TESTPATT;
            fi ;
        else 
            if [ "$NOCOT" = true ] ; then
                python3 test.py -noCoT -modelSize $MODELSIZE -dataset $DATASET -promptType $PROMPTTYPE -bestPromptType $BESTPROMPTTYPE -train $TRAIN -test $TEST -isTrainDir -isTestDir -trainPattern $TRAINPATT -testPattern $TESTPATT;
            else 
                python3 test.py -modelSize $MODELSIZE -dataset $DATASET -promptType $PROMPTTYPE -bestPromptType $BESTPROMPTTYPE -train $TRAIN -test $TEST -isTrainDir -isTestDir -trainPattern $TRAINPATT -testPattern $TESTPATT;
            fi ;
        fi ;
    else 
        if [ "$SELFCONSISTENCY" = true ] ; then
            if [ "$NOCOT" = true ] ; then
                python3 test.py -noCoT -selfConsistency -modelSize $MODELSIZE -dataset $DATASET -promptType $PROMPTTYPE -bestPromptType $BESTPROMPTTYPE -train $TRAIN -test $TEST -isTrainDir -trainPattern $TRAINPATT;
            else 
                python3 test.py -selfConsistency -modelSize $MODELSIZE -dataset $DATASET -promptType $PROMPTTYPE -bestPromptType $BESTPROMPTTYPE -train $TRAIN -test $TEST -isTrainDir -trainPattern $TRAINPATT;
            fi ;
        else
            if [ "$NOCOT" = true ] ; then
                python3 test.py -noCoT -modelSize $MODELSIZE -dataset $DATASET -promptType $PROMPTTYPE -bestPromptType $BESTPROMPTTYPE -train $TRAIN -test $TEST -isTrainDir -trainPattern $TRAINPATT;
            else 
                python3 test.py -modelSize $MODELSIZE -dataset $DATASET -promptType $PROMPTTYPE -bestPromptType $BESTPROMPTTYPE -train $TRAIN -test $TEST -isTrainDir -trainPattern $TRAINPATT;
            fi ;
        fi ;
    fi ;
else 
    if [ "$ISTESTDIR" = true ] ; then 
        if [ "$SELFCONSISTENCY" = true ] ; then
            if [ "$NOCOT" = true ] ; then
                python3 test.py -noCoT -selfConsistency -modelSize $MODELSIZE -dataset $DATASET -promptType $PROMPTTYPE -bestPromptType $BESTPROMPTTYPE -train $TRAIN -test $TEST -isTestDir -testPattern $TESTPATT;
            else 
                python3 test.py -selfConsistency -modelSize $MODELSIZE -dataset $DATASET -promptType $PROMPTTYPE -bestPromptType $BESTPROMPTTYPE -train $TRAIN -test $TEST -isTestDir -testPattern $TESTPATT;
            fi ;
        else
            if [ "$NOCOT" = true ] ; then
                python3 test.py -noCoT -modelSize $MODELSIZE -dataset $DATASET -promptType $PROMPTTYPE -bestPromptType $BESTPROMPTTYPE -train $TRAIN -test $TEST -isTestDir -testPattern $TESTPATT;
            else 
                python3 test.py -modelSize $MODELSIZE -dataset $DATASET -promptType $PROMPTTYPE -bestPromptType $BESTPROMPTTYPE -train $TRAIN -test $TEST -isTestDir -testPattern $TESTPATT;
            fi ;
        fi ;
    else
        if [ "$SELFCONSISTENCY" = true ] ; then 
            if [ "$NOCOT" = true ] ; then
                python3 test.py -noCoT -selfConsistency -modelSize $MODELSIZE -dataset $DATASET -promptType $PROMPTTYPE -bestPromptType $BESTPROMPTTYPE -train $TRAIN -test $TEST ;
            else 
                python3 test.py -selfConsistency -modelSize $MODELSIZE -dataset $DATASET -promptType $PROMPTTYPE -bestPromptType $BESTPROMPTTYPE -train $TRAIN -test $TEST ;
            fi ;
        else 
            if [ "$NOCOT" = true ] ; then
                python3 test.py -noCoT -modelSize $MODELSIZE -dataset $DATASET -promptType $PROMPTTYPE -bestPromptType $BESTPROMPTTYPE -train $TRAIN -test $TEST ;
            else 
                python3 test.py -modelSize $MODELSIZE -dataset $DATASET -promptType $PROMPTTYPE -bestPromptType $BESTPROMPTTYPE -train $TRAIN -test $TEST ;
            fi ;
        fi ;
    fi ;
fi