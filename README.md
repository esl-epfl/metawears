# MetaWearS: A Shortcut in Wearable Systems Lifecycle with Only a Few Shots

This repository contains the source code for the paper "MetaWearS: A Shortcut in Wearable Systems Lifecycle with Only a Few Shots".

## Introduction

Wearable systems provide continuous health monitoring and can lead to early detection of potential health issues. However, the lifecycle of wearable systems faces several challenges. First, effective model training for new wearable devices requires substantial labeled data from various subjects collected directly by the wearable. Second, subsequent model updates require further extensive labeled data for retraining. Finally, frequent model updating on the wearable device can decrease the battery life in long-term data monitoring.

Addressing these challenges, in this paper, we propose MetaWearS, a meta-learning method to reduce the amount of initial data collection required. Moreover, our approach incorporates a prototypical updating mechanism, simplifying the update process by modifying the class prototype rather than retraining the entire model. We explore the performance of MetaWearS in two case studies, namely, the detection of epileptic seizures and the detection of atrial fibrillation (AF).

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{amirshahi2024metawears,
      title={MetaWearS: A Shortcut in Wearable Systems Lifecycle with Only a Few Shots}, 
      author={Alireza Amirshahi and Maedeh H.Toosi and Siamak Mohammadi and Stefano Albini and Pasquale Davide Schiavone and Giovanni Ansaloni and Amir Aminifar and David Atienza},
      year={2024},
      eprint={2408.01988},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Repository Structure

```
.
├── notebooks/
│   ├── cross_dataset_result_analysis.ipynb  # Notebook for analyzing the results of the epileptic seizure detection case study.
│   └── data/
│       ├── results_meta_transfer.csv # Results data used in the notebook.
│       └── results_noisy.csv         # Results data for noise analysis.
├── scripts/
│   ├── ours_vs_maml.sh      # Shell script to run experiments.
│   └── run_script.sh        # Shell script to run experiments.
├── src/
│   ├── configs/
│   │   └── config.yml       # Configuration file for the experiments.
│   ├── experiments/         # Main experiment scripts.
│   │   ├── few_shot_train.py
│   │   ├── maml_metawears.py
│   │   ├── ...
│   │   └── inference.py
│   ├── utils/               # Utility scripts.
│   └── ...
├── environment.yml          # Conda environment file.
└── README.md
```

## Getting Started

### Prerequisites

This project uses [Conda](https://docs.conda.io/en/latest/) to manage the environment.

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/esl-epfl/metawears.git
    cd metawears
    ```
2.  Create the Conda environment from the `environment.yml` file:
    ```bash
    conda env create -f environment.yml
    ```
3.  Activate the environment:
    ```bash
    conda activate metawears
    ```

## Usage

The MetaWearS framework is divided into two main phases, as described in the paper. The following sections explain how to run the code for each phase.

### Phase 1: Initial Training and Fine-tuning

This phase corresponds to the initial data collection and training of the model (Challenge 1 in the paper). It involves two steps: pre-training on a large base dataset and fine-tuning on a smaller target dataset.

1.  **Pre-training:** The model is first pre-trained on a large, publicly available dataset (e.g., TUSZ for epilepsy, PhysioNet Challenge 2017 for AF). This is done using the `few_shot_train.py` script. You can configure the dataset path and other parameters in `src/configs/config.yml`. To run the pre-training, you can use the `run_script.sh` as an example:
    ```bash
    bash scripts/run_script.sh
    ```
    This script will execute the `few_shot_train.py` script with the appropriate arguments for pre-training.

2.  **Fine-tuning:** After pre-training, the model is fine-tuned on a smaller, target-specific dataset (e.g., Siena for epilepsy, or a private wearable dataset for AF). This step adapts the model to the specific characteristics of the wearable device and the user. The fine-tuning process is also handled by the `few_shot_train.py` script, with different arguments.

For more details on the training methodology, please refer to Section 2.4 of the paper.

### Prototypical Networks

MetaWearS uses Prototypical Networks to classify new samples. In each episode, prototypes for each class are calculated as the average of the feature vectors of the support samples. A new sample is then classified based on its Euclidean distance to the class prototypes. This mechanism is explained in detail in Section 2.3 of the paper.

### Phase 2: Model Update

This phase corresponds to updating the model with new data from the end-user (Challenge 2 in the paper). Instead of retraining the entire model, MetaWearS updates the class prototypes using the new samples. This is a lightweight and energy-efficient process.

The `few_shot_train.py` script can also be used to simulate this update process. By providing a few new shots of data, the script will update the prototypes and evaluate the model's performance. For more details, please refer to Section 2.5 of the paper.

### Hardware Implementation

The paper also presents a hardware implementation of the proposed method on an energy-efficient and low-power open-source platform (X-HEEP). The code for the hardware implementation can be found in the `hw-inference` directory. This implementation demonstrates the feasibility of deploying MetaWearS on resource-constrained wearable devices. For more details on the hardware architecture and results, please refer to Section 2.6 of the paper.

### Analyzing the Results

The results of the epileptic seizure detection case study can be reproduced and analyzed using the `notebooks/cross_dataset_result_analysis.ipynb` notebook. This notebook loads the results from the CSV files in `notebooks/data` and generates the plots presented in the paper.