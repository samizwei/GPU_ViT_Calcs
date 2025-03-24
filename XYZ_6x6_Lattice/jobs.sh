#!/bin/bash

# Run multiple scripts sequentially on all 8 GPUs
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python ViT_6x6_d24_nl1_MultipleRules_Xavier_patch22.py
python ViT_6x6_d24_nl1_MultipleRules_Xavier_patch32.py
python ViT_6x6_d24_nl2_MultipleRules_Xavier_patch22.py
python ViT_6x6_d24_nl2_MultipleRules_Xavier.py

# Run the final script on only 2 GPUs
export CUDA_VISIBLE_DEVICES=0,1
python ViT_6x6_d24_nl3_MultipleRules_Xavier.py