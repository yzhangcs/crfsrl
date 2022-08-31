# crfsrl

![](https://img.shields.io/badge/SRL-yellowgreen) ![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2Fbea48441d74ad3c72e960de0b7208a673a6948d2%3Ffields%3DcitationCount)

<p align="center"><img width="454" alt="image" src="https://user-images.githubusercontent.com/18402347/139386770-a1ca94a6-76c0-4d4c-8965-34a505d59ad4.png"></p>

Yu Zhang, Qingrong Xia, Shilin Zhou, Yong Jiang, Zhenghua Li, Guohong Fu, Min Zhang. _Semantic Role Labeling as Dependency Parsing: Exploring Latent Tree Structures Inside Arguments_. 2021. [[arxiv](http://arxiv.org/abs/2110.06865)]

## Setup

The following packages should be installed:
* [`PyTorch`](https://github.com/pytorch/pytorch): >= 1.12.1
* [`Transformers`](https://github.com/huggingface/transformers): >= 4.2

Clone this repo recursively:
```sh
git clone https://github.com/yzhangcs/crfsrl.git --recursive
```

Run the following scripts to obtain the training data.
Please make sure [PTB](http://catalog.ldc.upenn.edu/LDC99T42) and [OntoNotes](https://catalog.ldc.upenn.edu/LDC2013T19) are available:
```sh
bash scripts/conll05.sh PTB=<path-to-ptb>             SRL=data
bash scripts/conll12.sh ONTONOTES=<path-to-ontonotes> SRL=data
```

## Run

Try the following commands to train first-order CRF and second-order CRF2o models:
```sh
# LSTM
# CRF
python -u crf.py   train -b -c configs/conll05.crf.srl.lstm.char-lemma.ini   -d 0 -f char lemma -p exp/conll05.crf.srl.lstm.char-lemma/model   --cache --binarize
# CRF2o
python -u crf2o.py train -b -c configs/conll05.crf2o.srl.lstm.char-lemma.ini -d 0 -f char lemma -p exp/conll05.crf2o.srl.lstm.char-lemma/model --cache --binarize
# BERT finetuning
# CRF
python -u crf.py   train -b -c configs/conll05.crf.srl.bert.ini   -d 0 -p exp/conll05.crf.srl.bert/model   --batch-size=2000 --encoder bert --bert bert-large-cased --cache --binarize
# CRF2o
python -u crf2o.py train -b -c configs/conll05.crf2o.srl.bert.ini -d 0 -p exp/conll05.crf2o.srl.bert/model --batch-size=2000 --encoder bert --bert bert-large-cased --cache --binarize
```
To do evaluation:
```sh
# end-to-end
python -u crf.py   evaluate -c configs/conll05.crf.srl.bert.ini   -d 0 -p exp/conll05.crf.srl.bert/model
# w/ gold predicates
python -u crf.py   evaluate -c configs/conll05.crf.srl.bert.ini   -d 0 -p exp/conll05.crf.srl.bert/model --prd
```
To make predictions:
```sh
python -u crf.py   predict  -c configs/conll05.crf.srl.bert.ini   -d 0 -p exp/conll05.crf.srl.bert/model
bash scripts/eval.sh pred=pred.conllu gold=data/conll05/test.conllu
```

## Contact

If you have any questions, feel free to contact me via [emails](mailto:yzhang.cs@outlook.com).
