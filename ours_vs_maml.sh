#!/bin/bash

# Commands generated from results_with_validation.csv

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 3 --validation_patients 5 --finetune_patients 3 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 3 --validation_patients 5 --finetune_patients 3 --base_learner_root ./output_metawears/ --excluded_patients 5 3 1 17 6 12

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 3 1 17 --validation_patients 5 --finetune_patients 3 1 17 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 3 1 17 --validation_patients 5 --finetune_patients 3 1 17 --base_learner_root ./output_metawears/ --excluded_patients 5 3 1 17 6 12

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 3 1 17 6 12 --validation_patients 5 --finetune_patients 3 1 17 6 12 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 3 1 17 6 12 --validation_patients 5 --finetune_patients 3 1 17 6 12 --base_learner_root ./output_metawears/ --excluded_patients 5 3 1 17 6 12

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 12 --validation_patients 7 --finetune_patients 12 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 12 --validation_patients 7 --finetune_patients 12 --base_learner_root ./output_metawears/ --excluded_patients 7 12 9 5 0 10

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 12 9 5 --validation_patients 7 --finetune_patients 12 9 5 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 12 9 5 --validation_patients 7 --finetune_patients 12 9 5 --base_learner_root ./output_metawears/ --excluded_patients 7 12 9 5 0 10

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 12 9 5 0 10 --validation_patients 7 --finetune_patients 12 9 5 0 10 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 12 9 5 0 10 --validation_patients 7 --finetune_patients 12 9 5 0 10 --base_learner_root ./output_metawears/ --excluded_patients 7 12 9 5 0 10

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 0 --validation_patients 7 --finetune_patients 0 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 0 --validation_patients 7 --finetune_patients 0 --base_learner_root ./output_metawears/ --excluded_patients 7 0 1 9 6 12

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 0 1 9 --validation_patients 7 --finetune_patients 0 1 9 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 0 1 9 --validation_patients 7 --finetune_patients 0 1 9 --base_learner_root ./output_metawears/ --excluded_patients 7 0 1 9 6 12

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 0 1 9 6 12 --validation_patients 7 --finetune_patients 0 1 9 6 12 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 0 1 9 6 12 --validation_patients 7 --finetune_patients 0 1 9 6 12 --base_learner_root ./output_metawears/ --excluded_patients 7 0 1 9 6 12

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 0 --validation_patients 12 --finetune_patients 0 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 0 --validation_patients 12 --finetune_patients 0 --base_learner_root ./output_metawears/ --excluded_patients 12 0 17 7 5 10

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 0 17 7 --validation_patients 12 --finetune_patients 0 17 7 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 0 17 7 --validation_patients 12 --finetune_patients 0 17 7 --base_learner_root ./output_metawears/ --excluded_patients 12 0 17 7 5 10

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 0 17 7 5 10 --validation_patients 12 --finetune_patients 0 17 7 5 10 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 0 17 7 5 10 --validation_patients 12 --finetune_patients 0 17 7 5 10 --base_learner_root ./output_metawears/ --excluded_patients 12 0 17 7 5 10

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 6 --validation_patients 12 --finetune_patients 6 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 6 --validation_patients 12 --finetune_patients 6 --base_learner_root ./output_metawears/ --excluded_patients 12 6 1 10 7 5

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 6 1 10 --validation_patients 12 --finetune_patients 6 1 10 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 6 1 10 --validation_patients 12 --finetune_patients 6 1 10 --base_learner_root ./output_metawears/ --excluded_patients 12 6 1 10 7 5

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 6 1 10 7 5 --validation_patients 12 --finetune_patients 6 1 10 7 5 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 6 1 10 7 5 --validation_patients 12 --finetune_patients 6 1 10 7 5 --base_learner_root ./output_metawears/ --excluded_patients 12 6 1 10 7 5

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 17 --validation_patients 3 --finetune_patients 17 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 17 --validation_patients 3 --finetune_patients 17 --base_learner_root ./output_metawears/ --excluded_patients 3 17 1 0 9 7

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 17 1 0 --validation_patients 3 --finetune_patients 17 1 0 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 17 1 0 --validation_patients 3 --finetune_patients 17 1 0 --base_learner_root ./output_metawears/ --excluded_patients 3 17 1 0 9 7

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 17 1 0 9 7 --validation_patients 3 --finetune_patients 17 1 0 9 7 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 17 1 0 9 7 --validation_patients 3 --finetune_patients 17 1 0 9 7 --base_learner_root ./output_metawears/ --excluded_patients 3 17 1 0 9 7

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 17 --validation_patients 1 --finetune_patients 17 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 17 --validation_patients 1 --finetune_patients 17 --base_learner_root ./output_metawears/ --excluded_patients 1 17 10 9 0 5

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 17 10 9 --validation_patients 1 --finetune_patients 17 10 9 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 17 10 9 --validation_patients 1 --finetune_patients 17 10 9 --base_learner_root ./output_metawears/ --excluded_patients 1 17 10 9 0 5

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 17 10 9 0 5 --validation_patients 1 --finetune_patients 17 10 9 0 5 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 17 10 9 0 5 --validation_patients 1 --finetune_patients 17 10 9 0 5 --base_learner_root ./output_metawears/ --excluded_patients 1 17 10 9 0 5

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 5 --validation_patients 10 --finetune_patients 5 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 5 --validation_patients 10 --finetune_patients 5 --base_learner_root ./output_metawears/ --excluded_patients 10 5 9 1 3 17

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 5 9 1 --validation_patients 10 --finetune_patients 5 9 1 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 5 9 1 --validation_patients 10 --finetune_patients 5 9 1 --base_learner_root ./output_metawears/ --excluded_patients 10 5 9 1 3 17

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 5 9 1 3 17 --validation_patients 10 --finetune_patients 5 9 1 3 17 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 5 9 1 3 17 --validation_patients 10 --finetune_patients 5 9 1 3 17 --base_learner_root ./output_metawears/ --excluded_patients 10 5 9 1 3 17

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 12 --validation_patients 0 --finetune_patients 12 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 12 --validation_patients 0 --finetune_patients 12 --base_learner_root ./output_metawears/ --excluded_patients 0 12 6 10 9 17

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 12 6 10 --validation_patients 0 --finetune_patients 12 6 10 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 12 6 10 --validation_patients 0 --finetune_patients 12 6 10 --base_learner_root ./output_metawears/ --excluded_patients 0 12 6 10 9 17

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 12 6 10 9 17 --validation_patients 0 --finetune_patients 12 6 10 9 17 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 12 6 10 9 17 --validation_patients 0 --finetune_patients 12 6 10 9 17 --base_learner_root ./output_metawears/ --excluded_patients 0 12 6 10 9 17

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 6 --validation_patients 9 --finetune_patients 6 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 6 --validation_patients 9 --finetune_patients 6 --base_learner_root ./output_metawears/ --excluded_patients 9 6 17 10 0 3

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 6 17 10 --validation_patients 9 --finetune_patients 6 17 10 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 6 17 10 --validation_patients 9 --finetune_patients 6 17 10 --base_learner_root ./output_metawears/ --excluded_patients 9 6 17 10 0 3

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 6 17 10 0 3 --validation_patients 9 --finetune_patients 6 17 10 0 3 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 6 17 10 0 3 --validation_patients 9 --finetune_patients 6 17 10 0 3 --base_learner_root ./output_metawears/ --excluded_patients 9 6 17 10 0 3

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 5 --validation_patients 3 --finetune_patients 5 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 5 --validation_patients 3 --finetune_patients 5 --base_learner_root ./output_metawears/ --excluded_patients 3 5 6 0 9 7

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 5 6 0 --validation_patients 3 --finetune_patients 5 6 0 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 5 6 0 --validation_patients 3 --finetune_patients 5 6 0 --base_learner_root ./output_metawears/ --excluded_patients 3 5 6 0 9 7

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 5 6 0 9 7 --validation_patients 3 --finetune_patients 5 6 0 9 7 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 5 6 0 9 7 --validation_patients 3 --finetune_patients 5 6 0 9 7 --base_learner_root ./output_metawears/ --excluded_patients 3 5 6 0 9 7

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 7 --validation_patients 1 --finetune_patients 7 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 7 --validation_patients 1 --finetune_patients 7 --base_learner_root ./output_metawears/ --excluded_patients 1 7 9 3 5 10

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 7 9 3 5 10 --validation_patients 1 --finetune_patients 7 9 3 5 10 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 7 9 3 5 10 --validation_patients 1 --finetune_patients 7 9 3 5 10 --base_learner_root ./output_metawears/ --excluded_patients 1 7 9 3 5 10

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 12 --validation_patients 0 --finetune_patients 12 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 12 --validation_patients 0 --finetune_patients 12 --base_learner_root ./output_metawears/ --excluded_patients 0 12 6 10 5 17

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 12 6 10 --validation_patients 0 --finetune_patients 12 6 10 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 12 6 10 --validation_patients 0 --finetune_patients 12 6 10 --base_learner_root ./output_metawears/ --excluded_patients 0 12 6 10 5 17

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 12 6 10 5 17 --validation_patients 0 --finetune_patients 12 6 10 5 17 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 12 6 10 5 17 --validation_patients 0 --finetune_patients 12 6 10 5 17 --base_learner_root ./output_metawears/ --excluded_patients 0 12 6 10 5 17

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 17 --validation_patients 12 --finetune_patients 17 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 17 --validation_patients 12 --finetune_patients 17 --base_learner_root ./output_metawears/ --excluded_patients 12 17 5 7 0 3

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 17 5 7 --validation_patients 12 --finetune_patients 17 5 7 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 17 5 7 --validation_patients 12 --finetune_patients 17 5 7 --base_learner_root ./output_metawears/ --excluded_patients 12 17 5 7 0 3

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 17 5 7 0 3 --validation_patients 12 --finetune_patients 17 5 7 0 3 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 17 5 7 0 3 --validation_patients 12 --finetune_patients 17 5 7 0 3 --base_learner_root ./output_metawears/ --excluded_patients 12 17 5 7 0 3

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 6 --validation_patients 10 --finetune_patients 6 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 6 --validation_patients 10 --finetune_patients 6 --base_learner_root ./output_metawears/ --excluded_patients 10 6 3 7 1 12

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 6 3 7 --validation_patients 10 --finetune_patients 6 3 7 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 6 3 7 --validation_patients 10 --finetune_patients 6 3 7 --base_learner_root ./output_metawears/ --excluded_patients 10 6 3 7 1 12

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 6 3 7 1 12 --validation_patients 10 --finetune_patients 6 3 7 1 12 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 6 3 7 1 12 --validation_patients 10 --finetune_patients 6 3 7 1 12 --base_learner_root ./output_metawears/ --excluded_patients 10 6 3 7 1 12

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 6 --validation_patients 3 --finetune_patients 6 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 6 --validation_patients 3 --finetune_patients 6 --base_learner_root ./output_metawears/ --excluded_patients 3 6 10 9 17 12

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 6 10 9 --validation_patients 3 --finetune_patients 6 10 9 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 6 10 9 --validation_patients 3 --finetune_patients 6 10 9 --base_learner_root ./output_metawears/ --excluded_patients 3 6 10 9 17 12

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 6 10 9 17 12 --validation_patients 3 --finetune_patients 6 10 9 17 12 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 6 10 9 17 12 --validation_patients 3 --finetune_patients 6 10 9 17 12 --base_learner_root ./output_metawears/ --excluded_patients 3 6 10 9 17 12

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 6 --validation_patients 3 --finetune_patients 6 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 6 --validation_patients 3 --finetune_patients 6 --base_learner_root ./output_metawears/ --excluded_patients 3 6 10 5 17 12

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 6 10 5 --validation_patients 3 --finetune_patients 6 10 5 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 6 10 5 --validation_patients 3 --finetune_patients 6 10 5 --base_learner_root ./output_metawears/ --excluded_patients 3 6 10 5 17 12

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 6 10 5 17 12 --validation_patients 3 --finetune_patients 6 10 5 17 12 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 6 10 5 17 12 --validation_patients 3 --finetune_patients 6 10 5 17 12 --base_learner_root ./output_metawears/ --excluded_patients 3 6 10 5 17 12

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 6 --validation_patients 7 --finetune_patients 6 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 6 --validation_patients 7 --finetune_patients 6 --base_learner_root ./output_metawears/ --excluded_patients 7 6 9 12 5 1

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 6 9 12 5 1 --validation_patients 7 --finetune_patients 6 9 12 5 1 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 6 9 12 5 1 --validation_patients 7 --finetune_patients 6 9 12 5 1 --base_learner_root ./output_metawears/ --excluded_patients 7 6 9 12 5 1

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 12 --validation_patients 1 --finetune_patients 12 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 12 --validation_patients 1 --finetune_patients 12 --base_learner_root ./output_metawears/ --excluded_patients 1 12 7 5 17 6

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 12 7 5 --validation_patients 1 --finetune_patients 12 7 5 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 12 7 5 --validation_patients 1 --finetune_patients 12 7 5 --base_learner_root ./output_metawears/ --excluded_patients 1 12 7 5 17 6

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 12 7 5 17 6 --validation_patients 1 --finetune_patients 12 7 5 17 6 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 12 7 5 17 6 --validation_patients 1 --finetune_patients 12 7 5 17 6 --base_learner_root ./output_metawears/ --excluded_patients 1 12 7 5 17 6

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 6 --validation_patients 9 --finetune_patients 6 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 6 --validation_patients 9 --finetune_patients 6 --base_learner_root ./output_metawears/ --excluded_patients 9 6 7 17 1 12

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 6 7 17 --validation_patients 9 --finetune_patients 6 7 17 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 6 7 17 --validation_patients 9 --finetune_patients 6 7 17 --base_learner_root ./output_metawears/ --excluded_patients 9 6 7 17 1 12

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 6 7 17 1 12 --validation_patients 9 --finetune_patients 6 7 17 1 12 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 6 7 17 1 12 --validation_patients 9 --finetune_patients 6 7 17 1 12 --base_learner_root ./output_metawears/ --excluded_patients 9 6 7 17 1 12

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 12 --validation_patients 5 --finetune_patients 12 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 12 --validation_patients 5 --finetune_patients 12 --base_learner_root ./output_metawears/ --excluded_patients 5 12 17 9 7 0

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 12 17 9 --validation_patients 5 --finetune_patients 12 17 9 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 12 17 9 --validation_patients 5 --finetune_patients 12 17 9 --base_learner_root ./output_metawears/ --excluded_patients 5 12 17 9 7 0

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 12 17 9 7 0 --validation_patients 5 --finetune_patients 12 17 9 7 0 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 12 17 9 7 0 --validation_patients 5 --finetune_patients 12 17 9 7 0 --base_learner_root ./output_metawears/ --excluded_patients 5 12 17 9 7 0

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 6 --validation_patients 17 --finetune_patients 6 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 6 --validation_patients 17 --finetune_patients 6 --base_learner_root ./output_metawears/ --excluded_patients 17 6 9 5 10 7

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 6 9 5 --validation_patients 17 --finetune_patients 6 9 5 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 6 9 5 --validation_patients 17 --finetune_patients 6 9 5 --base_learner_root ./output_metawears/ --excluded_patients 17 6 9 5 10 7

python src/few_shot_train.py --finetune --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --epochs 25 --patients 6 9 5 10 7 --validation_patients 17 --finetune_patients 6 9 5 10 7 --base_learner_root ./output_metawears/
python src/few_shot_train.py --eval --experiment_root ./output_metawears --siena_data_dir ./input/siena/ --patients 6 9 5 10 7 --validation_patients 17 --finetune_patients 6 9 5 10 7 --base_learner_root ./output_metawears/ --excluded_patients 17 6 9 5 10 7