#!/bin/bash

data_root='/home/dk58319/shared/hdd_ext/nvme1/public/vision/classification/'
testsets=$1
arch=RN50
# arch=ViT-B/16
bs=64
ctx_init=a_photo_of_a

python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu 3 \
--tpt --ctx_init ${ctx_init}