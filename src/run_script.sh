#!/bin/bash

# Set the arguments
manual_seed_values=("1")
# manual_seed_values=("1")
num_support_val_values=("1" "3" "5" "10" "15" "20")
# num_support_val_values=("1")
# Original list
original_list=(0 1 3 5 6 7 9 10 12 17)
# Loop for four times
for iteration in {1..20}; do
    echo "Iteration $iteration:"
# Shuffle the list
shuffled_list=($(shuf -e "${original_list[@]}"))

# Select the first three elements
p1=${shuffled_list[0]}
p2=${shuffled_list[1]}
p3=${shuffled_list[2]}
p4=${shuffled_list[3]}

# Loop over manual_seed values
for seed in "${manual_seed_values[@]}"; do
    # Loop over num_support_val values
    python few_shot_train.py --finetune --patients "$p2" --validation_patients "$p1" --finetune_patients "$p2"  --epochs 50
    python few_shot_train.py --finetune --patients "$p2" "$p3" --validation_patients "$p1" --finetune_patients "$p2" "$p3" --epochs 50
    python few_shot_train.py --finetune --patients "$p2" "$p3" "$p4" --validation_patients "$p1" --finetune_patients "$p2" "$p3" "$p4" --epochs 50

    python few_shot_train.py --manual_seed "$seed" --num_support_val 5 --patients "$p2" --finetune_patients "$p2" --excluded_patients "$p1" "$p2" "$p3" "$p4"
    python few_shot_train.py --manual_seed "$seed" --num_support_val 5 --patients "$p2" "$p3" --finetune_patients "$p2" "$p3" --excluded_patients "$p1" "$p2" "$p3" "$p4"
    python few_shot_train.py --manual_seed "$seed" --num_support_val 5 --patients "$p2" "$p3"  "$p4" --finetune_patients "$p2" "$p3"  "$p4"  --excluded_patients "$p1" "$p2" "$p3" "$p4"


    for num_support in "${num_support_val_values[@]}"; do
        # Run the Python script with the given arguments

        python few_shot_train.py --manual_seed "$seed" --num_support_val "$num_support" --patients "$p2" "$p3" --finetune_patients "$p2" --excluded_patients "$p1" "$p2" "$p3" "$p4"
        python few_shot_train.py --manual_seed "$seed" --num_support_val "$num_support" --patients "$p2" "$p4" --finetune_patients "$p2" --excluded_patients "$p1" "$p2" "$p3" "$p4"

        python few_shot_train.py --manual_seed "$seed" --num_support_val "$num_support" --patients "$p2" "$p3" "$p4" --finetune_patients "$p2" --excluded_patients "$p1" "$p2" "$p3" "$p4"

        
         python few_shot_train.py --manual_seed "$seed" --num_support_val "$num_support" --patients "$p1" --finetune_patients "$p2" --excluded_patients "$p1" "$p2" "$p3" "$p4" --skip_finetune
         python few_shot_train.py --manual_seed "$seed" --num_support_val "$num_support" --patients "$p2" --finetune_patients "$p2" --excluded_patients "$p1" "$p2" "$p3" "$p4" --skip_finetune
         python few_shot_train.py --manual_seed "$seed" --num_support_val "$num_support" --patients "$p3" --finetune_patients "$p2" --excluded_patients "$p1" "$p2" "$p3" "$p4" --skip_finetune
         python few_shot_train.py --manual_seed "$seed" --num_support_val "$num_support" --patients "$p4" --finetune_patients "$p2" --excluded_patients "$p1" "$p2" "$p3" "$p4" --skip_finetune

         python few_shot_train.py --manual_seed "$seed" --num_support_val "$num_support" --patients "$p1" "$p2" "$p3" --finetune_patients "$p1" "$p2" "$p3" --excluded_patients "$p1" "$p2" "$p3" "$p4" --skip_finetune
         python few_shot_train.py --manual_seed "$seed" --num_support_val "$num_support" --patients "$p1" "$p2" "$p4" --finetune_patients "$p1" "$p2" "$p3" --excluded_patients "$p1" "$p2" "$p3" "$p4" --skip_finetune
         python few_shot_train.py --manual_seed "$seed" --num_support_val "$num_support" --patients "$p1" "$p3" "$p4" --finetune_patients "$p1" "$p2" "$p3" --excluded_patients "$p1" "$p2" "$p3" "$p4" --skip_finetune
         python few_shot_train.py --manual_seed "$seed" --num_support_val "$num_support" --patients "$p2" "$p3" "$p4" --finetune_patients "$p1" "$p2" "$p3" --excluded_patients "$p1" "$p2" "$p3" "$p4" --skip_finetune
    done

     python few_shot_train.py --finetune --validation_patients "$p1" --finetune_patients "$p2" --epochs 50 --skip_base_learner
     python few_shot_train.py --finetune --validation_patients "$p1" --finetune_patients "$p2" "$p3" --epochs 50 --skip_base_learner
     python few_shot_train.py --finetune --validation_patients "$p1" --finetune_patients "$p2" "$p3" "$p4" --epochs 50 --skip_base_learner

     python few_shot_train.py --manual_seed "$seed" --num_support_val 20 --patients "$p2" --finetune_patients "$p2" --excluded_patients "$p1" "$p2" "$p3" "$p4"  --skip_base_learner
     python few_shot_train.py --manual_seed "$seed" --num_support_val 20 --patients "$p2" "$p3" --finetune_patients "$p2" "$p3" --excluded_patients "$p1" "$p2" "$p3" "$p4" --skip_base_learner
     python few_shot_train.py --manual_seed "$seed" --num_support_val 20 --patients  "$p2" "$p3"  "$p4" --finetune_patients "$p2" "$p3"  "$p4"  --excluded_patients "$p1" "$p2" "$p3" "$p4" --skip_base_learner

done

done

