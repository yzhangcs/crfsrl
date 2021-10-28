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

echo "Collecting files"
# cat ${CONLL_ONTONOTES}/conll-formatted-ontonotes-5.0/data/train/data/english/annotations/*/*/*/*.gold_conll >> ${CONLL12}/train.english.v5_gold_conll
# cat ${CONLL_ONTONOTES}/conll-formatted-ontonotes-5.0/data/development/data/english/annotations/*/*/*/*.gold_conll >> ${CONLL12}/dev.english.v5_gold_conll
# cat ${CONLL_ONTONOTES}/conll-formatted-ontonotes-5.0/data/conll-2012-test/data/english/annotations/*/*/*/*.gold_conll >> ${CONLL12}/test.english.v5_gold_conll

TRAIN=${CONLL12}/train.english.v5_gold_conll
DEV=${CONLL12}/dev.english.v5_gold_conll
TEST=${CONLL12}/test.english.v5_gold_conll
TRAIN_ID=scripts/conll12/ids/english/coref/train.id
DEV_ID=scripts/conll12/ids/english/coref/development.id
TEST_ID=scripts/conll12/ids/english/coref/test.id

for file in TRAIN DEV TEST; do
  if [ ! -f ${CONLL12}/${file,,}.prop]; then
    echo "Converting ${!file} to prop format"
    id_file="$file"_ID
    rm ${CONLL12}/${file,,}.prop
    while IFS= read -r line; do
      id=$(awk -F ' ' '{print $1}' <<< "$line")
      if grep -Fq "$id" ${!id_file} ; then
        IFS=' ' read -ra cols <<< "$line"
        cols=("${cols[3]}" "${cols[6]}" "${cols[@]:11:${#cols[@]}-12}")
        cols=$(printf "\t%s" "${cols[@]}")
        cols=${cols:1}
        echo "$cols" >> ${CONLL12}/${file,,}.prop
      fi
    done < ${!file}
  fi
done
