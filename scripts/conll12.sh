#!/bin/bash

args=$@
for arg in $args; do
    eval "$arg"
done

DATA=~/.cache/supar/data

echo "ONTONOTES: ${ONTONOTES:=$DATA/ontonotes-release-5.0}"
echo "SRL:       ${SRL:=$DATA/srl}"

CONLL12=$SRL/conll12
CONLL_ONTONOTES=$DATA/conll-formatted-ontonotes-5.0

if [ ! -d $CONLL12 ]; then
  mkdir -p $CONLL12
fi

if [ ! -d $CONLL_ONTONOTES ]; then
  cp -r scripts/conll12/conll-formatted-ontonotes-5.0 $CONLL_ONTONOTES
  source activate py27
  cd $CONLL_ONTONOTES && ./conll-formatted-ontonotes-5.0/scripts/skeleton2conll.sh -D $ONTONOTES/data/files/data ./conll-formatted-ontonotes-5.0
  conda deactivate
fi

TRAIN=${CONLL_ONTONOTES}/conll-formatted-ontonotes-5.0/data/train
DEV=${CONLL_ONTONOTES}/conll-formatted-ontonotes-5.0/data/development
TEST=${CONLL_ONTONOTES}/conll-formatted-ontonotes-5.0/data/conll-2012-test
TRAIN_ID=scripts/conll12/ids/english/coref/train.id
DEV_ID=scripts/conll12/ids/english/coref/development.id
TEST_ID=scripts/conll12/ids/english/coref/test.id

for file in TRAIN DEV TEST; do
  if [ ! -f ${CONLL12}/${file,,}.conllu ]; then
    id_file="$file"_ID
    rm -f ${CONLL12}/${file,,}.prop
    if [ ! -f ${CONLL12}/${file,,}.gold_conll ]; then
      echo "Collecting files in ${!file} to ${CONLL12}/${file,,}.gold_conll"
      cat ${!file}/data/english/annotations/*/*/*/*.gold_conll > ${CONLL12}/${file,,}.gold_conll
    fi
    echo "Filtering annotations by id file"
    python scripts/conll12/filter.py --prop ${CONLL12}/${file,,}.gold_conll --fid ${!id_file} --file ${CONLL12}/${file,,}.prop
    echo "Converting ${CONLL12}/${file,,}.prop to conllu format"
    python scripts/prop2conllu.py --prop ${CONLL12}/${file,,}.prop --file ${CONLL12}/${file,,}.conllu
  fi
done
