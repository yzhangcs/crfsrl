#!/bin/bash

args=$@
for arg in $args; do
    eval "$arg"
done

DATA=~/.cache/supar/datasets

echo "ONTONOTES: ${ONTONOTES:=$DATA/ontonotes-release-5.0}"
echo "SRL:       ${SRL:=$DATA/srl}"

CONLL12=$SRL/conll12
CONLL_ONTONOTES=$DATA/conll-formatted-ontonotes-5.0

if [ ! -d $CONLL12 ]; then
  mkdir -p $CONLL12
fi

if [ ! -d $CONLL_ONTONOTES ]; then
  git clone https://github.com/yuchenlin/OntoNotes-5.0-NER-BIO.git $CONLL_ONTONOTES
  source ~/anaconda3/etc/profile.d/conda.sh
  conda activate py27
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
    echo "Collecting files in ${file,,} to ${CONLL12}/${file,,}.prop"
    id_file="$file"_ID
    rm ${CONLL12}/${file,,}.prop
    cat ${!file}/data/english/annotations/*/*/*/*.gold_conll | while IFS= read -r line; do
      id=$(awk -F ' ' '{print $1}' <<< "$line")
      if grep -Fq "$id" ${!id_file} ; then
        IFS=' ' read -ra cols <<< "$line"
        cols=("${cols[3]}" "${cols[6]}" "${cols[@]:11:${#cols[@]}-12}")
        cols=$(printf "\t%s" "${cols[@]}")
        cols=${cols:1}
        echo "$cols" >> ${CONLL12}/${file,,}.prop
      fi
    done
    echo "Converting ${CONLL12}/${file,,}.prop to the file of conllu format"
    python scripts/prop2conllu.py --prop ${CONLL12}/${file,,}.prop --file ${CONLL12}/${file,,}.conllu
  fi
done
