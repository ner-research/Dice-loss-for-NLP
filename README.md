# Dice Loss for NLP Tasks

> - Paper: [Dice Loss for Data-imbalanced NLP Tasks](https://arxiv.org/pdf/1911.02855.pdf) at ACL2020.
> - *This repository only contains NER task from [original github codes](https://github.com/ShannonAI/dice_loss_for_NLP).

## Setup

- Environments: `Python 3.6.9+` / `Pytorch 1.7.1` / `ubuntu GPU machine with CUDA 10.1`

```bash 
$ virtualenv -p /usr/bin/python3.6 venv
$ source venv/bin/activate
$ pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install -r requirements.txt
```

1. Download BERT Model Checkpoints (https://github.com/google-research/bert#pre-trained-models)

```
bash scripts/download_ckpt.sh <path-to-unzip-tf-bert-checkpoints> <model-name>
```

2. Convert original TensorFlow checkpoints for BERT to a PyTorch saved file

```
bash scripts/prepare_ckpt.sh <path-to-unzip-tf-bert-checkpoints>
```

## Named Entity Recognition 

For NER, we use MRC-NER model as the backbone. Processed datasets and model architecture can be found [here](https://arxiv.org/pdf/1910.11476.pdf). 

### Train

- Please run `scripts/<ner-datdaset-name>/bert_<loss-type>.sh` to train and evaluate on the dev set every `$val_check_interval` epoch.
- After training, the task trainer evaluates on the test set with the best checkpoint.
- The variable `<ner-dataset-name>` should take the value of `[ner_enontonotes5, ner_zhmsra, ner_zhonto4]`.
- The variable `<loss-type>` should take the value of `[focal, dice]` which denotes fine-tuning `BERT` with `focal loss`, `dice loss` , respectively. 

#### For English CoNLL03,

* Run `scritps/ner_enconll03/bert_focal.sh`. After training, you will get 93.08 Span-F1 on the test set. 

* Run `scripts/ner_enconll03/bert_dice.sh`. After training, you will get 93.21 Span-F1 on the test set.

#### For English OntoNotes5,

* Run `scripts/ner_enontonotes5/bert_focal.sh`. After training, you will get 91.12 Span-F1  on the test set.  

* Run `scripts/ner_enontonotes5/bert_dice.sh`. After training, you will get 92.01 Span-F1  on the test set. 

### Evaluate

To evaluate a model checkpoint, please run
```bash
CUDA_VISIBLE_DEVICES=0 python3 ${REPO_PATH}/tasks/mrc_ner/evaluate.py \
--gpus="1" \
--path_to_model_checkpoint $OUTPUT_DIR/epoch=2.ckpt
```

## References

- Original github code: https://github.com/ShannonAI/dice_loss_for_NLP
- xxiaoyali [AT] gmail.com OR xiaoya_li [AT] shannonai.com 

```tex 
@article{li2019dice,
  title={Dice loss for data-imbalanced NLP tasks},
  author={Li, Xiaoya and Sun, Xiaofei and Meng, Yuxian and Liang, Junjun and Wu, Fei and Li, Jiwei},
  journal={arXiv preprint arXiv:1911.02855},
  year={2019}
}
```
