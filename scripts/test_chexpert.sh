#!/bin/bash

data_root='/home/dk58319/private/CheXzero/data/test'
chest_weight_root='/home/dk58319/private/CheXzero/checkpoints/chexzero_weights/best_128_0.0002_original_15000_0.859.pt'

testsets=$1
arch=RN50
# arch=ViT-B/16

bs=64
ctx_init=a_photo_of_a

python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu 3 \
--tpt --ctx_init ${ctx_init} --chest ${chest_weight_root}