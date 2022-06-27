#!/bin/bash

args=$@
for arg in $args; do
    eval "$arg"
done

DATA=~/.cache/supar/data

echo "PTB: ${PTB:=$DATA/LDC1999T42}"
echo "SRL: ${SRL:=$DATA/srl}"

CONLL05=$SRL/conll05

if [ ! -d $CONLL05 ]; then
  mkdir -p $CONLL05
fi

if [ ! -d ${CONLL05}/conll05st-release ]; then
  echo "Downloading ConLL05 files"
  wget http://www.lsi.upc.edu/~srlconll/conll05st-release.tar.gz -O ${CONLL05}/conll05st-release.tar.gz
  wget http://www.lsi.upc.edu/~srlconll/conll05st-tests.tar.gz   -O ${CONLL05}/conll05st-tests.tar.gz
  tar xf ${CONLL05}/conll05st-release.tar.gz -C ${CONLL05}
  tar xf ${CONLL05}/conll05st-tests.tar.gz -C ${CONLL05}
fi
if [ ! -d $SRL/srlconll-1.1 ]; then
  echo "Downloading SRLEVAL scripts"
  wget http://www.lsi.upc.edu/~srlconll/srlconll-1.1.tgz -O $SRL/srlconll-1.1.tgz
  tar xf $SRL/srlconll-1.1.tgz -C $SRL
fi
export PERL5LIB="$SRL/srlconll-1.1/lib:$PERL5LIB"
export PATH="$SRL/srlconll-1.1/bin:$PATH"

TRAIN=train
DEV=devel
TEST=test.wsj
BROWN=test.brown
TRAIN_SECTIONS=(02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21)
DEV_SECTIONS=(24)
TEST_SECTIONS=(23)
BROWN_SECTIONS=(01 02 03)
for dataset in TRAIN DEV; do
  if [ ! -f "${CONLL05}/${dataset,,}.prop" ]; then
    echo "Generating the ${dataset,,} data"
    mkdir -p "${CONLL05}/conll05st-release/${!dataset}/words"
    SECTIONS="$dataset"_SECTIONS[@]
    for s in ${!SECTIONS}; do
      printf "$s "
      cat $PTB/PARSED/MRG/WSJ/$s/* | wsj-removetraces.pl | wsj-to-se.pl -w 1 | awk '{print $1}' | gzip > ${CONLL05}/conll05st-release/${!dataset}/words/${!dataset}.$s.words.gz
      zcat ${CONLL05}/conll05st-release/${!dataset}/words/${!dataset}.$s.words.gz > /tmp/$$.words
      zcat ${CONLL05}/conll05st-release/${!dataset}/props/${!dataset}.$s.props.gz > /tmp/$$.props
      paste -d '\t' /tmp/$$.words /tmp/$$.props | gzip > /tmp/$$.section.$s.gz
    done
    printf "\nSaving files to ${dataset,,}.prop\n"
    zcat /tmp/$$.section* > ${CONLL05}/${dataset,,}.prop && rm -f /tmp/$$*
  fi
done
for dataset in TEST BROWN; do
  if [ ! -f ${CONLL05}/${dataset,,}.prop ]; then
    echo "Generating the ${dataset,,} data"
    zcat ${CONLL05}/conll05st-release/${!dataset}/words/${!dataset}.words.gz > /tmp/$$.words
    zcat ${CONLL05}/conll05st-release/${!dataset}/props/${!dataset}.props.gz > /tmp/$$.props
    echo "Saving files to ${dataset,,}.prop"
    paste -d '\t' /tmp/$$.words /tmp/$$.props > ${CONLL05}/${dataset,,}.prop && rm -f /tmp/$$*
  fi
done

echo "Extracting evaluation files"
zcat ${CONLL05}/conll05st-release/devel/props/devel.24.props.gz > ${CONLL05}/conll05.dev.props.gold.txt
zcat ${CONLL05}/conll05st-release/test.wsj/props/test.wsj.props.gz > ${CONLL05}/conll05.test.props.gold.txt
zcat ${CONLL05}/conll05st-release/test.brown/props/test.brown.props.gz > ${CONLL05}/conll05.test.brown.props.gold.txt

rm -f ${CONLL05}/conll05st-release.tar.gz
rm -f ${CONLL05}/conll05st-tests.tar.gz

for dataset in train dev test brown; do
  echo "Converting ${CONLL05}/$dataset.prop to conllu format"
  python scripts/prop2conllu.py --prop ${CONLL05}/$dataset.prop --file ${CONLL05}/$dataset.conllu
done
