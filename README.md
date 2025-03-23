This repository is for the paper: "MetaWearS: A Shortcut in Wearable System Lifecycles with Only a Few Shots"
The source code can be found in the "src" directory. In particular, src/few_shot_train.py represents the training, finetuning and evaluation procedure in MetaWearS. 


## How to use this code

### Dataset Preparation

1.  **Download Datasets:** Download the desired wearable datasets (e.g., UCI HAR, WISDM, etc.) and place them in a designated directory (e.g., `data/`).
2.  **Preprocessing:** Each dataset might require specific preprocessing steps (e.g., normalization, windowing). Refer to the dataset's documentation or existing scripts for guidance.
3.  **Data Format:** Ensure the data is formatted appropriately for the `few_shot_train.py` script. Typically, this involves organizing the data into training, validation, and testing sets, with each set containing sensor readings and corresponding labels.
4. **Configuration:** Update the configuration file (e.g., `config.yaml`) with the correct paths to your preprocessed datasets.

### Running `few_shot_train.py`

1.  **Dependencies:** Ensure you have all the necessary Python packages installed. You can typically install them using `pip install -r requirements.txt`.
2.  **Configuration:** Modify the `config.yaml` file to specify the desired parameters for training, finetuning, and evaluation. This includes:
    *   Dataset paths
    *   Model architecture
    *   Hyperparameters (learning rate, batch size, etc.)
    *   Few-shot settings (number of shots, number of ways)
    *   Evaluation metrics
3.  **Execution:** Run the `few_shot_train.py` script from the command line:

    ```bash
    python src/few_shot_train.py --config config.yaml
    ```

    *   Replace `config.yaml` with the path to your configuration file if it's different.
4. **Output:** The script will output the training, finetuning and evaluation results to the console and save them in the specified output directory.
