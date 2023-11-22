#!/bin/bash

data_root='/home/dk58319/shared/hdd_ext/nvme1/public/vision/classification/'
coop_weight='/home/dk58319/shared/hdd_ext/nvme1/donghyun/CoOp_Weights/rn50_ep50_16shots/nctx4_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50'
testsets=$1
arch=RN50
# arch=ViT-B/16
bs=64

python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu 0 \
--tpt --load ${coop_weight}