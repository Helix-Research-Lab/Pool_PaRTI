#!/bin/bash



sbatch --export=POOLING_METHOD=cls,START=0,END=95 run_serial.sh
sbatch --export=POOLING_METHOD=mean_pooled,START=0,END=95 run_serial.sh
sbatch --export=POOLING_METHOD=max_pooled,START=0,END=95 run_serial.sh
sbatch --export=POOLING_METHOD=PR_contact_prune_topk_no_enh,START=0,END=95 run_serial.sh







