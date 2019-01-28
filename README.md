State-LSTM for Relation Extraction
==========

This repo contains the *PyTorch* code for the [State-LSTM](https://arxiv.org/abs/1808.09101) in relation extraction task.  

Difference between this repo and the [code](https://github.com/freesunshine0316/nary-grn) released by author
- this repo is more clean, while the author's code including many unrelated code for Machine Reading, Sequence Tagging.
- this repo is implemented in adjacency matrix manner.
- this is using PyTorch, the author's code is using Tensorflow.

The scaffold is forked from [this repo](https://github.com/qipeng/gcn-over-pruned-trees/blob/master/model/gcn.py).

See below for an overview of the model architecture:

![State LSTM Architecture](fig/state-lstm.png "State-LSTM Architecture")

## Requirements

- Python 3 (tested on 3.6.6)
- PyTorch (tested on 0.4.1)
- tqdm
- unzip, wget (for downloading only)

## Preparation

The code requires that you have access to the [TACRED dataset](https://nlp.stanford.edu/projects/tacred/) (LDC license required). 

First, download and unzip GloVe vectors from the Stanford NLP group website, with:
```
chmod +x download.sh; ./download.sh
```

Then prepare vocabulary and initial word vectors with:
```
python prepare_vocab.py dataset/tacred dataset/vocab --glove_dir dataset/glove
```

This will write vocabulary and word vectors as a numpy matrix into the dir `dataset/vocab`.

## Training

To train a state LSTM neural network model, run:
```
bash train_sl.sh 0
```

Model checkpoints and logs will be saved to `./saved_models/00`.

For details on the use of other parameters, such as the time_steps, please refer to `train.py`.

## Evaluation

To run evaluation on the test set, run:
```
python eval.py saved_models/00 --dataset test
```

This will use the `best_model.pt` file by default. Use `--model checkpoint_epoch_10.pt` to specify a model checkpoint file.

## Retrain

Reload a pretrained model and finetune it, run:
```
python train.py --load --model_file saved_models/01/best_model.pt --optim sgd --lr 0.001
```

## Citation

```
@article{song2018n,
  title={N-ary relation extraction using graph state LSTM},
  author={Song, Linfeng and Zhang, Yue and Wang, Zhiguo and Gildea, Daniel},
  journal={arXiv preprint arXiv:1808.09101},
  year={2018}
}
```

## License

All work contained in this package is licensed under the Apache License, Version 2.0. See the included LICENSE file.
