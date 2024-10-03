#!/bin/bash

#sbatch train_one_model.sh --run_no 1 --pooling_method cls --learning_rate 0.0001 --max_LR 0.001 --num_epochs 40 --weight_decay 0.01 --dropout_rate 0.05 --hidden_dim 1024 --clip_value 1 --exponent_class_weight 1

## python -u train.py --run_no 999 --pooling_method cls --hyperparam_group 84 --rapid_debug_mode --early_stopping --num_epochs 1 --rapid_debug_mode 2>&1 | tee debugOutputs/output1.txt



#sbatch train_one_model.sh --run_no 1 --pooling_method cls --hyperparam_group 1 --early_stopping --num_epochs 1000 --patience_es 1000


################################# HYPERPARAM RUNS

# sbatch --export=POOLING_METHOD=clsPR_attn_max_prune_1inout_no_enh,START=0,END=1 run_serial.sh
#######################################################################################################################################################

#sbatch --export=POOLING_METHOD=clsPR_attn_max_prune_1inout_no_enh,START=12,END=35 run_serial.sh
#sbatch --export=POOLING_METHOD=clsPR_attn_max_prune_1inout_no_enh,START=49,END=71 run_serial.sh

#sbatch --export=POOLING_METHOD=cls_pooled,START=8,END=14 run_serial_efficient.sh
#sbatch --export=POOLING_METHOD=cls_pooled,START=25,END=29 run_serial_efficient.sh
#sbatch --export=POOLING_METHOD=cls_pooled,START=33,END=44 run_serial_efficient.sh

#sbatch --export=POOLING_METHOD=mean_pooled,START=11,END=14 run_serial_efficient.sh
#sbatch --export=POOLING_METHOD=mean_pooled,START=28,END=29 run_serial_efficient.sh
#sbatch --export=POOLING_METHOD=mean_pooled,START=40,END=44 run_serial_efficient.sh

##sbatch --export=POOLING_METHOD=max_pooled,START=0,END=14 run_serial_efficient.sh
##sbatch --export=POOLING_METHOD=max_pooled,START=15,END=29 run_serial_efficient.sh
##sbatch --export=POOLING_METHOD=max_pooled,START=30,END=44 run_serial_efficient.sh

#sbatch --export=POOLING_METHOD=PR_contact_prune_topk_no_enh,START=7,END=14 run_serial_efficient.sh
#sbatch --export=POOLING_METHOD=PR_contact_prune_topk_no_enh,START=20,END=29 run_serial_efficient.sh
#sbatch --export=POOLING_METHOD=PR_contact_prune_topk_no_enh,START=35,END=44 run_serial_efficient.sh


#sbatch --export=POOLING_METHOD=cls_pooled,START=44,END=59 run_serial_efficient.sh
#sbatch --export=POOLING_METHOD=cls_pooled,START=60,END=74 run_serial_efficient.sh
#sbatch --export=POOLING_METHOD=cls_pooled,START=75,END=89 run_serial_efficient.sh

#sbatch --export=POOLING_METHOD=mean_pooled,START=44,END=59 run_serial_efficient.sh
#sbatch --export=POOLING_METHOD=mean_pooled,START=60,END=74 run_serial_efficient.sh
#sbatch --export=POOLING_METHOD=mean_pooled,START=75,END=80 run_serial_efficient.sh

#sbatch --export=POOLING_METHOD=max_pooled,START=44,END=59 run_serial_efficient.sh
#sbatch --export=POOLING_METHOD=max_pooled,START=60,END=74 run_serial_efficient.sh
#sbatch --export=POOLING_METHOD=max_pooled,START=75,END=89 run_serial_efficient.sh

#sbatch --export=POOLING_METHOD=PR_contact_prune_topk_no_enh,START=44,END=59 run_serial_efficient.sh
#sbatch --export=POOLING_METHOD=PR_contact_prune_topk_no_enh,START=60,END=74 run_serial_efficient.sh
#sbatch --export=POOLING_METHOD=PR_contact_prune_topk_no_enh,START=75,END=89 run_serial_efficient.sh

sbatch --export=POOLING_METHOD=cls_pooled,START=90,END=106 run_serial_efficient.sh
sbatch --export=POOLING_METHOD=cls_pooled,START=107,END=125 run_serial_efficient.sh

sbatch --export=POOLING_METHOD=PR_contact_prune_topk_no_enh,START=90,END=106 run_serial_efficient.sh
sbatch --export=POOLING_METHOD=PR_contact_prune_topk_no_enh,START=107,END=125 run_serial_efficient.sh

sbatch --export=POOLING_METHOD=mean_pooled,START=90,END=106 run_serial_efficient.sh
sbatch --export=POOLING_METHOD=mean_pooled,START=107,END=125 run_serial_efficient.sh

sbatch --export=POOLING_METHOD=max_pooled,START=90,END=106 run_serial_efficient.sh
sbatch --export=POOLING_METHOD=max_pooled,START=107,END=125 run_serial_efficient.sh







