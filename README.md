# crfsrl

Code for "[Semantic Role Labeling as Dependency Parsing: Exploring Latent Tree Structures Inside Arguments](http://arxiv.org/abs/2110.06865)".

## Setup

The following packages should be installed:
* [`SuPar`](https://github.com/yzhangcs/parser): == 1.1.3
* [`PyTorch`](https://github.com/pytorch/pytorch): >= 1.7
* [`Transformers`](https://github.com/huggingface/transformers): >= 4.0

Run following scripts to obtain the training data.
Please make sure [PTB files](http://catalog.ldc.upenn.edu/LDC99T42) are available:
```sh
bash scripts/conll05.sh PTB=<path-to-your-ptb-file> SRL=data
```

## Run

Try the following commands to train first-order CRF and second-order CRF2o models:
```sh
# LSTM
# CRF
python -u crf.py   train -b -c configs/conll05.crf.srl.lstm.char-lemma.ini   -d 0 -f char lemma -p exp/conll05.crf.srl.lstm.char-lemma/model
# CRF2o
python -u crf2o.py train -b -c configs/conll05.crf2o.srl.lstm.char-lemma.ini -d 0 -f char lemma -p exp/conll05.crf2o.srl.lstm.char-lemma/model
# BERT finetuning
# CRF
python -u crf.py   train -b -c configs/conll05.crf.srl.bert.ini   -d 0 -p exp/conll05.crf.srl.bert/model   --batch-size=1000 --encoder bert --bert bert-large-cased 
# CRF2o
python -u crf2o.py train -b -c configs/conll05.crf2o.srl.bert.ini -d 0 -p exp/conll05.crf2o.srl.bert/model --batch-size=1000 --encoder bert --bert bert-large-cased
```

## Contact

If you have any questions, feel free to contact me via [emails](mailto:yzhang.cs@outlook.com).