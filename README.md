# AQ_Europe
This repository contains the codebase and experiments for the Bachelor thesis project:
“Active Learning for Air Quality Monitoring in Europe” (xixellonja/AQ_Europe/thesis_air_quality.pdf).

The project investigates how active learning can be applied to reduce annotation costs when training a mlp classifier using partially labeled EEA validated data.

# Key Features:
- Multi-class classification of PM2.5 pollution levels using a neural network
- Use of real-world air quality data from the EEA (European Environment Agency)
- Evaluation of active learning strategies:
    - BADGE (Gradient Embedding Sampling)
    - TypiClust with COP-KMeans
- Comparison to baseline random sampling under class imbalance. 
- Visualization of model performance, query behavior, and learning curves across multiple seeds.

NOTE: This codebase was developed over a 4-week period. Due to time constraints, the current codebase contains some redundant logic and non-modular scripts. A full refactoring is planned to enhance it.

# Instructions
- Step 1: To run the main project across all seeds, please use: sbatch --output="./log/log_all.out" --error="./log/log_all.err" run_all.sh
NOTE: After running, all plots used in the paper + accuracy values as "label_acc_xxx.txt." files,  should be available for all seeds in /home/stud/kellezi)


- Step 2: To calculate the model's average accuracy across all seeds (Baselne vs AL), please use: sbatch --output="./log/log_all__pred_accuracy.out" --error="./log/log_all_pred_accuracy.err" pred_accuracy_all.sh
     - The accuracy values across all seeds are computed in Step 1 and manually inserted from the output files "label_acc_xxx.txt".
     - The label and overall prediction accuracy values (avg and std deviation) should be available in the log file under "/home/stud/kellezi/log/log_all__pred_accuracy.out"


- Step 3 (last experiment 5.1.3) : To query instances from the entire EEA (validated + unvalidated) data with BADGE, please use: sbatch --output="./log/log_42_badge.out" --error="./log/log_42_badge.err" badge_query_all_data.sh The label and category counts of the query are logged in ./log/log_42_badge.out
