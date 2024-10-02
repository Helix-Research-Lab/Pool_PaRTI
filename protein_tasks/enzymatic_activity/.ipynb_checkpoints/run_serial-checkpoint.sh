#!/bin/bash
#SBATCH --job-name=hpt_enzyme
##SBATCH --partition=#ANON
#SBATCH --gres gpu:1
#SBATCH --time=1-23:59:00
#SBATCH -C GPU_MEM:16GB



conda activate poolparti
#ml gcc/10.1.0
#ml cuda/12.1.1

nvidia-smi

pooling_method=${POOLING_METHOD}
start=${START}
end=${END}

# Loop from 6 to 90 for combination numbers
#for i in {0..95}; do
for ((i=start; i<=end; i++)); do
    echo "Running training for combination number $i"
    # Construct the run name dynamically based on the loop index
    

    python train.py \
        --run_no $i \
        --pooling_method $pooling_method \
        --hyperparam_group $i \
        --early_stopping \
        
        
    if [ $? -ne 0 ]; then
        echo "First script failed for combination $i, exiting..."
    fi


    
done

#sbatch --export=POOLING_METHOD=cls run_serial.sh






