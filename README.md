
This repository contains the implementation and experimental data for the paper titled "Mining Verdict Boundaries for Neural Network Verification". The contents are being organized and updated regularly. 




<details>
<summary><strong>Installation Guide</strong></summary>
<p>
**Note: BMINER is developed based on alpha-beta-CROWN. Please follow the steps below to install and set up the alpha-beta-CROWN environment first.**

## 1. Initialize Submodules
First, initialize and update the git submodules. Run the following commands in your terminal:

`git submodule update --init`

## 2. Create the alpha-beta-CROWN Virtual Environment

Next, set up the virtual environment for alpha-beta-CROWN. You can use the following commands:
```
# Remove the old environment, if necessary.
conda deactivate; conda env remove --name alpha-beta-crown1
# install all dependents into the alpha-beta-crown environment
conda env create -f alpha-beta-CROWN/complete_verifier/environment.yaml --name alpha-beta-crown1
# activate the environment
conda activate alpha-beta-crown1
```

## 3. Move code files

The third step is to move all files from the `code` directory to `alpha-beta-CROWN/complete_verifier`. You can use the following command:

`cp -r code/* alpha-beta-CROWN/complete_verifier/`

</p>
</details>

