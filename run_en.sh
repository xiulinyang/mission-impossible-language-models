#!/bin/bash

# Accept a language argument
LANGUAGE=$1
VOCAB=$2
GPU=$3
PERMUTATION=$4

echo "PERMUTATION: $PERMUTATION"
echo "LANGUAGE: ${LANGUAGE,,}"
echo "Combined: ${PERMUTATION}_${LANGUAGE,,}"

# Default value if no argument is provided
if [ -z "$LANGUAGE" ]; then
	  echo "No language specified. Using 'EN' as default."
	    LANGUAGE="EN"
fi

cd data


python perturb.py ${PERMUTATION}_${LANGUAGE,,} $LANGUAGE train
python perturb.py ${PERMUTATION}_${LANGUAGE,,} $LANGUAGE dev
python perturb.py ${PERMUTATION}_${LANGUAGE,,} $LANGUAGE test

cd ..
cd training

bash prepare_training.sh ${PERMUTATION}_${LANGUAGE,,} $LANGUAGE 53 randinit

cd ..
cd mistral

conda init bash
source ~/.bashrc
conda deactivate
conda activate mistral

CUDA_VISIBLE_DEVICES=$GPU python3 train.py --config conf/train_${PERMUTATION}_${LANGUAGE,,}_${LANGUAGE}_randinit_seed53.yaml --nnodes 1 --nproc_per_node 1 --training_arguments.fp16 true --training_arguments.warmup_steps 120 --training_arguments.max_steps 1200
    
cd ..
cd perplexities
conda activate mission
CUDA_VISIBLE_DEVICES=$GPU python perplexities_exp.py ${PERMUTATION}_${LANGUAGE,,} ${PERMUTATION}_${LANGUAGE,,} $LANGUAGE 53 randinit $VOCAB
