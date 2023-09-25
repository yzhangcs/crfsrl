<div align="center">

# Semantic Role Labeling as Dependency Parsing: Exploring Latent Tree Structures Inside Arguments

<div>
  <a href='https://yzhang.site/' target='_blank'><b>Yu Zhang</b></a><sup>1</sup>&emsp;
  <a href='https://kirosummer.github.io/' target='_blank'>Qingrong Xia</a><sup>1,2</sup>&emsp;
  <a href='https://github.com/zsLin177' target='_blank'>Shilin Zhou</a><sup>1</sup>&emsp;
  <a href='https://jiangyong.site/' target='_blank'>Yong Jiang</b></a><sup>3</sup>&emsp;
  <a href='https://web.suda.edu.cn/ghfu/' target='_blank'>Guohong Fu</a><sup>1</sup>&emsp;
  <a href='https://zhangminsuda.github.io/' target='_blank'>Min Zhang</a><sup>1</sup>&emsp;
</div>
<div><sup>1</sup>Institute of Artificial Intelligence, School of Computer Science and Technology, Soochow University, Suzhou, China</div>
<div><sup>2</sup>Huawei Cloud, China</div>
<div><sup>3</sup>DAMO Academy, Alibaba Group, China</div>

<div>
<h4>

[![conf](https://img.shields.io/badge/COLING%202022-orange?style=flat-square)](https://aclanthology.org/2022.coling-1.370/)
[![arxiv](https://img.shields.io/badge/arXiv-2110.06865-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2110.06865)
[![citation](https://img.shields.io/badge/dynamic/json?label=citation&query=citationCount&url=https%3A%2F%2Fapi.semanticscholar.org%2Fgraph%2Fv1%2Fpaper%2F64332d61dfef5ac685500a238b8a79d75152c164%3Ffields%3DcitationCount&style=flat-square)](https://www.semanticscholar.org/paper/Semantic-Role-Labeling-as-Dependency-Parsing%3A-Tree-Zhang-Xia/64332d61dfef5ac685500a238b8a79d75152c164)
![python](https://img.shields.io/badge/python-%3E%3D%203.7-pybadges.svg?logo=python&style=flat-square)

</h4>
</div>

<p align="center">
<img width="400" alt="image" src="https://user-images.githubusercontent.com/18402347/191160039-2024f0d5-54c5-4cb7-81a5-ba90d3335dfe.png">
</p>
</div>

## Citation

If you are interested in our work, please cite
```bib
@inproceedings{zhang-etal-2022-semantic,
  title     = {Semantic Role Labeling as Dependency Parsing: Exploring Latent Tree Structures inside Arguments},
  author    = {Zhang, Yu  and
               Xia, Qingrong  and
               Zhou, Shilin  and
               Jiang, Yong  and
               Fu, Guohong  and
               Zhang, Min},
  booktitle = {Proceedings of COLING},
  year      = {2022},
  url       = {https://aclanthology.org/2022.coling-1.370},
  address   = {Gyeongju, Republic of Korea},
  publisher = {International Committee on Computational Linguistics},
  pages     = {4212--4227}
}
```

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
