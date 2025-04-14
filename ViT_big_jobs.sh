#!/bin/bash

# # Run multiple scripts sequentially on all 8 GPUs
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0,1,2,3

python /scratch/samiz/GPU_ViT_Calcs/XYZ_8x8_Lattice_ViT/patching_xy44_signstructure/ViT_8x8_d24_nl1_SS_First.py

python /scratch/samiz/GPU_ViT_Calcs/XYZ_8x8_Lattice_ViT/patching_xy44_signstructure/ViT_8x8_d24_nl1_SS_Second.py


python /scratch/samiz/GPU_ViT_Calcs/XYZ_8x8_Lattice_ViT/patching_xy22/ViT_8x8_d24_nl1_patch22_Second.py


python /scratch/samiz/GPU_ViT_Calcs/XYZ_10x10_Lattice_ViT/patching_xy55/ViT_10x10_d32_nl1_First.py
# python /scratch/samiz/GPU_ViT_Calcs/XYZ_8x8_Lattice_ViT/patching_xy22/ViT_8x8_d24_nl1_patch22_First.py

# python /scratch/samiz/GPU_ViT_Calcs/XYZ_8x8_Lattice_ViT/patching_xy44_signstructure/ViT_8x8_d24_nl1_First.py

# python /scratch/samiz/GPU_ViT_Calcs/XYZ_8x8_Lattice_ViT/patching_xy44_signstructure/ViT_8x8_d24_nl1_SS_Second.py

# python /scratch/samiz/GPU_ViT_Calcs/XYZ_10x10_Lattice_ViT/patching_xy55/ViT_10x10_d24_nl1_First.py


# python vit_d24_nl1_5050_Xavier_patch22.py
# python vit_d24_nl1_5050_Xavier_patch44.py
# python vit_d24_nl2_MultipleRulesXavier.py
# python vit_d24_nl3_5050_Xavier.py
