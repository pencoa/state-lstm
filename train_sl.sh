#!/bin/bash

SAVE_ID=$1
python train.py --id $SAVE_ID --seed 0 --optim adam --lr 0.001 --lr_decay 1 --log_step 20 --rnn_hidden 150 --num_epoch 200 --epoch_counter 100 --batch_size 50 --pooling max --mlp_layers 2 --pooling_l2 0.003 --pool_type piece
