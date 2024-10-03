>📋  A template README.md for code accompanying a Machine Learning paper

# Pool PaRTI: A PageRank-based Pooling Method for Robust Protein Sequence Representation in Deep Learning

This repository is the official implementation of [Pool PaRTI: A PageRank-based Pooling Method for Robust Protein Sequence Representation in Deep Learning]. 

![graphical_summary_of_Pool_PaRTI](https://github.com/user-attachments/assets/de0b97e9-069c-43ea-9a0d-da8fe80fe3cc)

## Requirements and environment setup

To install requirements:

```setup
conda env create --file environment.yml
```

For a list of proteins with UniProt accessions, we provide the code to create fasta files (./create_fasta_files), to create token embeddings (./token_embedding_generator), and to create sequence embedding through Pool PaRTI and the three baselines (./pooling_generator). Once the sequence embeddings are generated, to run training in any of the tasks we have benchmarked on, follow the instructions below:

## Embedding Generation

To activate the conda, run this command:

```activation
conda activate poolparti
```

Once the conda environment is activated, the following steps need to be taken to generate a sequence embedding vector.
1) fasta files should be created for the sequence,
2) token embeddings & attention map generation with the protein language model of interest (ESM-2 650M or protBERT in this repo)
3) applying Pool PaRTI on the token embeddings and the attention maps

Below, we provide the code to run to achieve each of the steps

### 1. Fasta file generation

Go to the creating_fasta_files/ directory. Create a .txt (e.g., accessions.txt) file that has a list of uniprot accessions for the proteins you wish to create embeddings for, one accession per line. See an example of this in creating_fasta_files/test_accessions.txt

Then, run
```fasta run
python fetch_uniprot_sequences_individual.py --accessions accessions.txt --output_dir ./sequences
```

This will create individual FASTA files in the creating_fasta_files/sequences directory, each named after the corresponding UniProt accession number.

### 2. ESM-2 or protBERT token embedding and attention map generation
To generate token embeddings and attention maps using either the ESM-2 or ProtBERT model, use the provided scripts in the token_embedding_generator/ directory.

First, navigate to the directory: 
```
cd token_embedding_generator/
```

#### using ESM-2 model

To generate embeddings using the ESM-2 model, run:
```
python process_fasta_files.py --model esm --input_dir ../creating_fasta_files/sequences --output_dir ./esm_embeddings

```

To generate embeddings using the ProtBERT model, run:
```
python process_fasta_files.py --model protbert --input_dir ../creating_fasta_files/sequences --output_dir ./protbert_embeddings
```

### 3. Sequence embedding generation with Pool PaRTI
To generate sequence embeddings using Pool PaRTI, apply the pooling method to the token embeddings and attention maps generated in the previous step.

Navigate to the pooling_generator/ directory:

```
cd ../pooling_generator/
```
Run the following command:

```
python pooled_sequence_generator.py --path_token_emb PATH_TO_TOKEN_EMBEDDINGS \
                                        --path_attention_layers PATH_TO_ATTENTION_MATRICES \
                                        [--output_dir OUTPUT_DIRECTORY] \
                                        [--generate_all]
```
where you replace the PATH_TO_TOKEN_EMBEDDINGS and PATH_TO_ATTENTION_MATRICES like the following example:

```
python pooled_sequence_generator.py --path_token_emb ../esm_embeddings/esm/representation_matrices/P07550.pt \
                                    --path_attention_layers ./esm_embeddings/esm/attention_matrices_mean_max_perLayer/P07550.pt \
                                    --output_dir ./sequence_embeddings                                  
```

This script will apply the Pool PaRTI method to the token embeddings and attention maps to generate a single embedding vector for each protein sequence. By including the --generate_all flag, you can also generate mean pooling, max pooling, sum pooling and CLS token pooling as alternatives for comparison.


## Experiments that show Pool PaRTI embedding performance boost 
### Training

To activate the conda, run this command:

```activation
conda activate poolparti
```

To train the model(s) in the paper, go to the respective directory under protein_tasks directory and run these commands:

```train
chmod 775 run_parallel.sh
./run_parallel.sh
```

The run_parallel.sh scripts queses a number of jobs through sbatch, but the job scheduling can be edited to conform to the user's environment. The underlying bash script is environment-agnostic. 

### Evaluation

The training script also automatically generates evaluation metrics and predicted raw output files on an independent test set. The results can be analyzed by running the cells in the notebooks in the respective task directory.

## Citation

If you find this repository or method useful in your research, please cite:

@article{YourArticle,
  title={Pool PaRTI: A PageRank-based Pooling Method for Robust Protein Sequence Representation in Deep Learning},
  author={Your Name and Co-authors},
  journal={Journal Name or arXiv preprint},
  year={2023},
  url={link_to_paper}
}

## License
This project is licensed under the Apache-2.0 license - see the LICENSE file for details.

## Acknowledgement
We would like to thank the members of the Altman Lab for useful discussions.
