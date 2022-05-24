#!/bin/bash

args=$@
for arg in $args; do
    eval "$arg"
done

DATA=~/.cache/supar/data

echo "pred:    ${pred:=pred.prop}"
echo "gold:    ${gold:=gold.prop}"
echo "srleval: ${srleval:=$DATA/srl/srlconll-1.1}"

if [ ! -d $srleval ]; then
  echo "Downloading SRLEVAL scripts"
  wget http://www.lsi.upc.edu/~srlconll/srlconll-1.1.tgz -O $(dirname $srleval)/srlconll-1.1.tgz
  tar xf $(dirname $srleval)/srlconll-1.1.tgz -C $(dirname $srleval)
fi
export PERL5LIB="${srleval}/lib:$PERL5LIB"
export PATH="${srleval}/bin:$PATH"


if [[ $pred == *.conllu ]]; then
  echo "Converting $pred to prop format"
  python scripts/conllu2prop.py --conllu $pred --file $pred.prop
  pred=$pred.prop
fi
if [[ $gold == *.conllu ]]; then
  echo "Converting $gold to prop format"
  python scripts/conllu2prop.py --conllu $gold --file $gold.prop
  gold=$gold.prop
fi

P=$(perl ${srleval}/bin/srl-eval.pl $pred $gold | grep Overall | awk -F ' ' '{print $6}')
R=$(perl ${srleval}/bin/srl-eval.pl $gold $pred | grep Overall | awk -F ' ' '{print $6}')
printf  "%.2f %.2f %.2f\n" $P $R "$(bc -l <<< "2*$P*$R/($P+$R)")"
