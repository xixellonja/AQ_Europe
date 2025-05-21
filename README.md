# AQ_Europe
Bachelor Thesis Project:: Active Learning for Air Quality in Europe

Step 1: sbatch --output="./log/log_all.out" --error="./log/log_all.err" run_all.sh
To run the main project across all seeds, please use: sbatch --output="./log/log_all.out" --error="./log/log_all.err" run_all.sh
NOTE: After running, all plots used in the paper + accuracy values as "label_acc_xxx.txt." files,  should be available for all seeds in /home/stud/kellezi)


Step 2. sbatch --output="./log/log_all__pred_accuracy.out" --error="./log/log_all_pred_accuracy.err" pred_accuracy_all.sh
To calculate the model's average accuracy across all seeds (Baselne vs AL), please use: sbatch --output="./log/log_all__pred_accuracy.out" --error="./log/log_all_pred_accuracy.err" pred_accuracy_all.sh
NOTE:
- The accuracy values across all seeds are computed in Step 1 and manually inserted from the output files "label_acc_xxx.txt".
- The label and overall prediction accuracy values (avg and std deviation) should be available in the log file under "/home/stud/kellezi/log/log_all__pred_accuracy.out"


Step 3 (last experiment 5.1.3) : sbatch --output="./log/log_42_badge.out" --error="./log/log_42_badge.err" badge_query_all_data.sh
To query instances from the entire EEA (validated + unvalidated) data with BADGE, please use: sbatch --output="./log/log_42_badge.out" --error="./log/log_42_badge.err" badge_query_all_data.sh
The label and category counts of the query are logged in ./log/log_42_badge.out
