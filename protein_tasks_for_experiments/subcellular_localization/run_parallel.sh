#!/bin/bash


sbatch --export=POOLING_METHOD=cls_pooled,START=0,END=35 run_serial_efficient.sh

sbatch --export=POOLING_METHOD=PR_contact_prune_topk_no_enh,START=0,END=35 run_serial_efficient.sh

sbatch --export=POOLING_METHOD=mean_pooled,START=90,END=106 run_serial_efficient.sh
sbatch --export=POOLING_METHOD=mean_pooled,START=107,END=125 run_serial_efficient.sh

sbatch --export=POOLING_METHOD=max_pooled,START=90,END=106 run_serial_efficient.sh
sbatch --export=POOLING_METHOD=max_pooled,START=107,END=125 run_serial_efficient.sh







