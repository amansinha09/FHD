import os
import subprocess
import itertools

# Define hyperparameter search space
hyperparams = {
    "which_loss": ["balancedsoftmax", "equalizationloss"], # "softmax", "wsoftmax",], # "focalloss", "classbalancedloss", "balancedsoftmax", "equalizationloss", "ldamloss"],
    "epochs": [50],
    "batch_size": [16, 32, 64],
    "learning_rate": [1e-5, 3e-5, 5e-5],
    "input": ["tt"], #["title", "text", "tt"],
    "model_name": ["bert-base-uncased"], # "roberta-base", "distilbert-base-uncased", "albert-base-v2", "google/electra-base-discriminator"],
    "data_dir": ["./data"],
    "seed": [7897,45689, 78907],
    "maxlen": [512],
    "patience": [3]
}

# Generate all combinations of hyperparameters
keys, values = zip(*hyperparams.items())
hyperparam_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]


script_name = "main_loss.py"

# Create directory to store results
#results_dir = "./results"
#os.makedirs(results_dir, exist_ok=True)

# Loop through combinations and execute script
#count =0

name2plm = {"bert-base-uncased":"bert", "roberta-base":"roberta", "distilbert-base-uncased":"distilbert", "albert-base-v2":"alberta", "google/electra-base-discriminator":"electra"}

i=0
message = f"""
#!/bin/bash

# Add the SBATCH directives
#SBATCH --job-name={name2plm[hyperparams["model_name"][0]]}.{i+1}
#SBATCH --account=project_2007780
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH -o %x.log
#SBATCH --mail-type=ALL

# Load modules and set variables
module load pytorch;

expname={name2plm[hyperparams["model_name"][0]]}.{i+1}

logdir=/scratch/project_2007780/amasi/FHD/shells

export HF_HOME=/scratch/project_2005099/members/mickusti/cerberus/hf



"""


for cc, combination in enumerate(hyperparam_combinations):
    # Construct the command
    
    cmd = [
        "python", script_name
    ] + [
        f"--{key} {value}" for key, value in combination.items()
    ]
    with open(f'{name2plm[hyperparams["model_name"][0]]}.{i+1}.sh', 'w') as fp:
        print(message, file=fp)
        print(' '.join(cmd), file=fp)
    i +=1
    #print()

print("TOTAL=", i)
