>📋  A template README.md for code accompanying a Machine Learning paper

# Pool PaRTI: A PageRank-based Pooling Method for Robust Protein Sequence Representation in Deep Learning

This repository is the official implementation of [Pool PaRTI: A PageRank-based Pooling Method for Robust Protein Sequence Representation in Deep Learning]. 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements and environment setup

To install requirements:

```setup
conda env create --file environment.yml
```

For a list of proteins with UniProt accessions, we provide the code to create fasta files (./create_fasta_files), to create token embeddings (./token_embedding_generator), and to create sequence embedding through Pool PaRTI and the three baselines (./pooling_generator). Once the sequence embeddings are generated, to run training in any of the tasks we have benchmarked on, follow the instructions below:

## Training

To activate the conda, run this command:

```activation
conda activate poolparto
```

To train the model(s) in the paper, go to the respective directory under protein_tasks directory and run these commands:

```train
chmod 775 run_parallel.sh
./run_parallel.sh
```

The run_parallel.sh scripts queses a number of jobs through sbatch, but the job scheduling can be edited to conform to the user's environment. The underlying bash script is environment-agnostic. 

## Evaluation

The training script also automatically generates evaluation metrics and predicted raw output files on an independent test set. The results can be analyzed by running the cells in the notebooks in the respective task directory.
