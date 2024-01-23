# AbPROP: Antibody Property Prediction

This repository contains the code for the AbPROP models presented in the ICML 2023 Computational Biology Workshop paper titled "AbPROP: Language and Graph Deep Learning for Antibody Property Prediction".

## Step 0 - Data Preparation

Before proceeding with the program, make sure you have all the experimental or predicted protein data files (in PDB format) for the sequences you want to test. These files should be aligned in a Multiple Sequence Alignment (MSA) format. Additionally, you will need the corresponding labels for each protein, as well as a predefined train/test split. If you have separate Heavy and Light chains, align them separately and then concatenate them to create the full MSA.

In this step, you need to prepare a protein dataframe with the following columns:

- `name`: sequence identifier
- `split`: train or holdout
- `light_msa`: aligned light chain
- `heavy_msa`: aligned heavy chain
- `msa`: concatenation of heavy and light MSA
- `target`: scalar or binary property
- `structure_path`: absolute path to the associated PDB file

For single chain prediction (e.g., vHH), only specify the `msa` column.

## Step 1 - Data Processing

To process the data, run the following command:

`python prepare_data.py -d <dataset_name> -t target --data-file <path_to_file_from_step_0> -c <"single" or "both"> -o jsons/`


This script will generate two JSON files, `proteins_<split>_<dataset>.json`, in the `jsons/` folder. These JSON files contain the processed data in a graph representation, which is ready to be used for training the model.

## Step 2 - Training

The `hp_tuning.py` script is available for hyperparameter tuning with cross-validation. Once your data is prepared, you can use this script to train the model. The script provides various options.

To train the sequence model (ablang + linear head) with 5-fold cross-validation and exploring all the default hyperparameters on a scalar dataset, open an interactive session on your favorite GPU and run the following command:

`python hp_tuning.py -o 1 -d psr -c both -n 50 -p y -k 5`


## Step 3 - Evaluation

After training the model with cross-validation for hyperparameter tuning, an ensemble of the k models with the highest combined accuracy will be saved for each combination of dataset and AbPROP model type. The ensemble will be saved in the `outputs/best_models/psr_linear/` directory, and the average validation score and hyperparameters will be saved in the `outputs/best_models/{dataset}_{model}/` directory.

To evaluate the ensemble predictions on the holdout data, we can use the `ensemble.py` script. This script requires the dataset to predict on, the number of models to ensemble (k from cross-validation), and the model to use ('linear', 'gvp', 'mifst', or 'gat'). Note that the holdout sizes are currently hardcoded in the script, so if you want to predict on a different holdout set, you need to modify the script.

Example usage of `ensemble.py`:

`python ensemble.py -d psr -h gvp -k 5`

