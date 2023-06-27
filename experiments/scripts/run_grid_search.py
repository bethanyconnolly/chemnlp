import json
import subprocess

from chemnlp.data_val.config import GridSearch
from chemnlp.utils import _get_all_combinations

# User-defined parameters
MULTINODE_RUNS = True
SBATCH_SCRIPT = (
    "experiments/scripts/sbatch_train_hf_multinode.sh"
    if MULTINODE_RUNS
    else "experiments/scripts/sbatch_train_hf.sh"
)

WANDB_GRID_GROUPNAME = "1B_smiles_gridsearch"
CONDA_ENV = "experiments/training-env"
CHEMNLP_FOLDER = "beth"

BASE_CONFIGS = ["1B_fine_tune.yml"]
GRID_PARAMETERS = GridSearch(
    data={"path": [
        "s3://llched-tokenised/1B_pubmed_smiles_train",
        "s3://llched-tokenised/1B_pubmed_smiles_valid_train",
        "s3://llched-tokenised/1B_pubmed_smiles_valid_lift_train",
        "s3://llched-tokenised/1B_pubmed_smiles_lift_train",
        ]},
    trainer={"learning_rate": [1e-4, 1e-5]},
)

if __name__ == "__main__":
    # Job submission loop
    for config_path in BASE_CONFIGS:
        # for each base configuration
        config_name = config_path.split(".")[0]
        all_possible_hyperparams = _get_all_combinations(GRID_PARAMETERS.dict())

        for i, overriding_params in enumerate(all_possible_hyperparams):
            # set checkpoint dir & wandb run name
            run_name = f"{config_name}_{i}"
            overriding_params["wandb"]["name"] = run_name
            overriding_params["wandb"]["group"] = WANDB_GRID_GROUPNAME
            overriding_params["trainer"][
                "output_dir"
            ] = f"/fsx/proj-chemnlp/experiments/checkpoints/finetuned/{WANDB_GRID_GROUPNAME}/{run_name}"
            # remove spaces for bash
            overriding_json = f"'{json.dumps(overriding_params)}'".replace(" ", "")

            # submit every combination of grid search parameters
            cmd = f"sbatch {SBATCH_SCRIPT} {CONDA_ENV} {CHEMNLP_FOLDER} {config_path} {overriding_json}"
            subprocess.run(cmd, shell=True)
